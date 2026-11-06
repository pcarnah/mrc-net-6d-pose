"""Usprobe dataset adapter with mono/stereo scene pairing.

Source layout (under ``cfg.DATASET_ROOT/usprobe``)::

    <split>/<scene:06d>/{gray,rgb}/<img:06d>.<ext>
    <split>/<scene:06d>/mask_visib/<img:06d>_<anno:06d>.png
    <split>/<scene:06d>/{scene_camera,scene_gt,scene_gt_info}.json
    models/obj_XXXXXX.ply, models_eval/obj_XXXXXX.ply

The scenes are recorded as consecutive left/right pairs: scene ``2k`` is the
left eye and scene ``2k+1`` the matched right eye of the same capture
(``train_df.csv`` / ``test_df.csv`` confirm the ``eye`` column).  A matched
pair shares the same image ids and the same object rotation; the right-eye
translation is the left-eye translation shifted by the rig baseline along x
(``t_r = t_l - [b, 0, 0]``, ``b`` ~ 50.04 mm for the d435i and ~18.07 mm for
the d405).

Modes (``DATASET_CONFIG['usprobe']['stereo_mode']``):

    mono   identical to ``BOP_Dataset``: one sample per (left scene, image,
           instance), targets in the left camera frame.
    stereo one sample per left instance that has a matching right annotation.
           ``__getitem__`` returns the left-eye PoseSample plus the right-eye
           crop inputs and the per-sample rig transform ``T_right_ref``
           (left -> right camera frame), mirroring ``stereobj_dataset``.

The matched scenes are validated while the index is built: the image-id sets
must agree and every paired annotation must correspond (same rotation, a
translation difference that is a pure x shift).  Mismatches are logged and the
offending instances are dropped, so a partially-corrupted pair still
contributes its valid frames.
"""
import os

import numpy as np
import torch
from tqdm import tqdm

import config as cfg
import utils
from bop_dataset import BOP_Dataset
from lib import data_utils as misc
from lib import file_io

import logging
logger = logging.getLogger(__name__)

# Pose-correspondence tolerances for a left/right annotation pair.  A genuine
# stereo pair has R_l ~= R_r and t_l - t_r ~= [b, 0, 0]; pairs whose frame
# indices do not align (e.g. the d405 scenes 108-143) violate both badly and
# are dropped.  This is a per-pair frame-numbering artifact, not a property of
# the camera (the camera only sets the baseline magnitude).
ROT_TOL = 1e-2          # max abs element difference of R_l vs R_r
TRANS_YZ_TOL_MM = 5.0   # max |t_l - t_r| in the y/z axes
BASELINE_RANGE_MM = (1.0, 200.0)


def pair_scenes(scenes):
    """Consecutive ``(left, right)`` scene pairs from a sorted scene list.

    Returns ``(pairs, leftover)`` where ``leftover`` is the trailing scene
    when the count is odd (``None`` otherwise).
    """
    pairs = [(scenes[i], scenes[i + 1]) for i in range(0, len(scenes) - 1, 2)]
    leftover = scenes[-1] if len(scenes) % 2 else None
    return pairs, leftover


def check_scene_pair(left_gt, right_gt):
    """Image ids present on only one side of a matched scene pair."""
    left_ids, right_ids = set(left_gt), set(right_gt)
    return (sorted(left_ids - right_ids, key=int),
            sorted(right_ids - left_ids, key=int))


def _pose_delta(left_anno, right_anno):
    """``(dR, dt)`` between a left and right annotation of the same object."""
    R_l = np.asarray(left_anno['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
    R_r = np.asarray(right_anno['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
    t_l = np.asarray(left_anno['cam_t_m2c'], dtype=np.float32).reshape(3)
    t_r = np.asarray(right_anno['cam_t_m2c'], dtype=np.float32).reshape(3)
    return R_l, t_l, R_r, t_r


def _pair_is_consistent(R_l, t_l, R_r, t_r):
    """True when a left/right annotation pair looks like a real stereo pair."""
    if np.abs(R_l - R_r).max() > ROT_TOL:
        return False
    dt = t_l - t_r
    if abs(float(dt[1])) > TRANS_YZ_TOL_MM or abs(float(dt[2])) > TRANS_YZ_TOL_MM:
        return False
    return BASELINE_RANGE_MM[0] <= abs(float(dt[0])) <= BASELINE_RANGE_MM[1]


def _orthonormalize(R):
    """Nearest proper rotation (SVD) to a nearly-orthonormal matrix.

    The stored ``cam_R_m2c`` matrices carry ~2e-4 of non-orthonormality, so
    ``R_r @ R_l.T`` is not a rotation even when R_l == R_r.  The model composes
    the right render with ``T_right_ref``'s rotation, which must be a proper
    rotation, so orthonormalize before forming the relative transform.
    """
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U = U.copy()
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def validate_stereo_pairs(dataset_name='usprobe', split='train'):
    """Scan every matched scene pair and report image/pose mismatches.

    Cheap and index-only (no quat labels, no image decode), so it is safe to
    call from a standalone check.  Returns a dict with per-pair issues and a
    summary count.
    """
    ds_cfg = cfg.DATASET_CONFIG[dataset_name]
    name_set = {'train': ds_cfg['train_set'],
                'finetune': ds_cfg.get('finetune_set', ds_cfg['train_set']),
                'test': ds_cfg['test_set']}[split]
    data_dir = os.path.join(cfg.DATASET_ROOT, dataset_name)

    report = {'pairs': 0, 'image_mismatch': [], 'pose_mismatch': [],
              'missing_scene': [], 'leftover': None, 'instances': 0,
              'valid_instances': 0}
    for img_type in name_set:
        split_dir = os.path.join(data_dir, img_type)
        scenes = sorted(s for s in os.listdir(split_dir) if s.startswith('00'))
        pairs, leftover = pair_scenes(scenes)
        if leftover is not None:
            report['leftover'] = leftover
        for left_scene, right_scene in pairs:
            report['pairs'] += 1
            right_dir = os.path.join(split_dir, right_scene)
            if not os.path.isdir(right_dir):
                report['missing_scene'].append((left_scene, right_scene))
                continue
            left_gt = file_io.load(
                os.path.join(split_dir, left_scene, 'scene_gt.json'))
            right_gt = file_io.load(os.path.join(right_dir, 'scene_gt.json'))
            missing_r, extra_r = check_scene_pair(left_gt, right_gt)
            if missing_r or extra_r:
                report['image_mismatch'].append(
                    (left_scene, right_scene, missing_r, extra_r))
            for img_id in sorted(set(left_gt) & set(right_gt), key=int):
                right_annos = right_gt[img_id]
                for left_anno in left_gt[img_id]:
                    report['instances'] += 1
                    match = next(
                        (a for a in right_annos
                         if a['obj_id'] == left_anno['obj_id']), None)
                    if match is None:
                        report['pose_mismatch'].append(
                            (left_scene, right_scene, img_id,
                             left_anno['obj_id'], 'no_right_anno'))
                        continue
                    R_l, t_l, R_r, t_r = _pose_delta(left_anno, match)
                    if _pair_is_consistent(R_l, t_l, R_r, t_r):
                        report['valid_instances'] += 1
                    else:
                        report['pose_mismatch'].append(
                            (left_scene, right_scene, img_id,
                             left_anno['obj_id'],
                             float(np.abs(R_l - R_r).max())))
    return report


class UsprobeDataset(BOP_Dataset):
    """Mono/stereo adapter for the usprobe BOP-layout dataset.

    Mono mode is exactly ``BOP_Dataset`` (this class only overrides the index
    builder and the read path, both of which delegate to ``super()``).
    """

    def _build_image_records(self):
        if self.stereo_mode == 'stereo':
            if not cfg.DATASET_CONFIG[self.dataset_name].get(
                    'stereo_scene_pairs', False):
                raise ValueError(
                    "usprobe stereo_mode='stereo' requires "
                    "'stereo_scene_pairs': True in DATASET_CONFIG")
            return self._build_stereo_records()
        return super()._build_image_records()

    # ------------------------------------------------------------------ #
    # Stereo index
    # ------------------------------------------------------------------ #
    def _build_stereo_records(self):
        """Image-level records where each annotation carries both eyes.

        Only left-scene instances with a consistent right counterpart are
        kept; the right eye's image/mask/bbox/intrinsics are attached to the
        left annotation so ``read_data_stereo`` can crop both eyes.
        """
        dataset_dicts = []
        for img_type in self.name_set:
            train_dir = os.path.join(self.data_dir, img_type)
            model_folder = os.path.join(
                self.data_dir, self.model_folders[img_type])
            scenes = sorted(
                s for s in os.listdir(train_dir) if s.startswith('00'))
            pairs, leftover = pair_scenes(scenes)
            if leftover is not None:
                logger.warning(
                    "usprobe stereo: odd scene count in %s, dropping %s",
                    img_type, leftover)

            image_counter = 0
            instance_counter = 0
            for left_scene, right_scene in tqdm(pairs, desc=img_type):
                left_dir = os.path.join(train_dir, left_scene)
                right_dir = os.path.join(train_dir, right_scene)
                if not os.path.isdir(right_dir):
                    logger.warning(
                        "usprobe stereo: missing right scene %s for %s",
                        right_scene, left_scene)
                    continue
                left_gt = file_io.load(
                    os.path.join(left_dir, "scene_gt.json"))
                right_gt = file_io.load(
                    os.path.join(right_dir, "scene_gt.json"))
                left_info = file_io.load(
                    os.path.join(left_dir, "scene_gt_info.json"))
                right_info = file_io.load(
                    os.path.join(right_dir, "scene_gt_info.json"))
                left_cam = file_io.load(
                    os.path.join(left_dir, "scene_camera.json"))
                right_cam = file_io.load(
                    os.path.join(right_dir, "scene_camera.json"))

                missing_r, extra_r = check_scene_pair(left_gt, right_gt)
                if missing_r or extra_r:
                    logger.warning(
                        "usprobe stereo: %s/%s image-set mismatch "
                        "(missing_right=%s extra_right=%s)",
                        left_scene, right_scene, missing_r, extra_r)

                scene_id = int(left_scene)
                for img_id_str in sorted(left_gt, key=int):
                    if img_id_str not in right_gt:
                        continue
                    img_id_int = int(img_id_str)
                    left_annos = left_gt[img_id_str]
                    right_annos = right_gt[img_id_str]
                    left_bboxes = left_info[img_id_str]
                    right_bboxes = right_info[img_id_str]

                    left_img = self._resolve_image_path(left_dir, img_id_int)
                    right_img = self._resolve_image_path(
                        right_dir, img_id_int)
                    cam_K_l = np.array(
                        left_cam[img_id_str]["cam_K"],
                        dtype=np.float32).reshape(3, 3)
                    cam_K_r = np.array(
                        right_cam[img_id_str]["cam_K"],
                        dtype=np.float32).reshape(3, 3)

                    view_insts = []
                    view_inst_count = dict()
                    used_right = set()
                    for anno_idx, anno_dict in enumerate(left_annos):
                        obj_id = anno_dict["obj_id"]
                        if obj_id not in self.dataset_id2cls:
                            continue
                        bbox_visib = left_bboxes[anno_idx]["bbox_visib"]
                        x1, y1, w, h = bbox_visib
                        if self.filter_invalid and (h <= 10 or w <= 10):
                            continue
                        visib_fract = left_bboxes[anno_idx]["visib_fract"]
                        if visib_fract < 0.10:
                            continue

                        # Match the right annotation of the same object.
                        r_idx = None
                        for cand, r_anno in enumerate(right_annos):
                            if cand in used_right:
                                continue
                            if r_anno["obj_id"] == obj_id:
                                r_idx = cand
                                break
                        if r_idx is None:
                            continue
                        used_right.add(r_idx)

                        R_l, t_l, R_r, t_r = _pose_delta(
                            anno_dict, right_annos[r_idx])
                        if not _pair_is_consistent(R_l, t_l, R_r, t_r):
                            continue

                        r_bbox = right_bboxes[r_idx]["bbox_visib"]
                        rx1, ry1, rw, rh = r_bbox
                        if rw <= 0 or rh <= 0:
                            continue
                        r_mask_file = os.path.join(
                            right_dir, "mask_visib/{:06d}_{:06d}.png".format(
                                img_id_int, r_idx))
                        l_mask_file = os.path.join(
                            left_dir, "mask_visib/{:06d}_{:06d}.png".format(
                                img_id_int, anno_idx))
                        if not os.path.exists(r_mask_file) \
                                or not os.path.exists(l_mask_file):
                            continue

                        model_path = os.path.join(
                            model_folder, 'obj_{:06d}.ply'.format(int(obj_id)))
                        mask_l = self._load_mask_rle(l_mask_file)
                        mask_r = self._load_mask_rle(r_mask_file)

                        if obj_id not in view_inst_count:
                            view_inst_count[obj_id] = 0
                        view_inst_count[obj_id] += 1

                        inst = {
                            'sub_dataset_folder': img_type,
                            # left eye (targets)
                            'image_file': left_img,
                            'mask_file': mask_l,
                            'model_file': model_path,
                            'bbox': bbox_visib,
                            'quat_file': os.path.join(
                                left_dir,
                                "quat_label/{:06d}_{:06d}.npy".format(
                                    img_id_int, anno_idx)),
                            'rotation': R_l,
                            'translation': t_l,
                            'intrinsics': cam_K_l,
                            # right eye (inputs only)
                            'image_file_right': right_img,
                            'mask_file_right': mask_r,
                            'bbox_right': r_bbox,
                            'intrinsics_right': cam_K_r,
                            'rotation_right': R_r,
                            'translation_right': t_r,
                            # meta
                            'scene_id': scene_id,
                            'im_id': img_id_int,
                            'obj_id': int(obj_id),
                        }
                        view_insts.append(inst)

                    if len(view_insts) == 0:
                        continue
                    record = {
                        "dataset_name": self.dataset_name,
                        "scene_id": scene_id,
                        "image_id": img_id_int,
                        "img_type": img_type,
                        "height": self.height,
                        "width": self.width,
                        "annotations": view_insts,
                        "obj_inst_count": view_inst_count,
                    }
                    dataset_dicts.append(record)
                    image_counter += 1
                    instance_counter += len(view_insts)

            print(img_type, ', stereo images: ', image_counter,
                  ', instances: ', instance_counter)
        return dataset_dicts

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def read_data(self, dataset_dict, idx):
        if self.stereo_mode == 'stereo':
            return self.read_data_stereo(dataset_dict, idx)
        return super().read_data(dataset_dict, idx)

    def _maybe_color_aug(self, image, mask):
        """Photometric augmentation, identical to ``BOP_Dataset.read_data``."""
        if (self.split in ('train', 'finetune')
                and np.random.rand() < self.COLOR_AUG_PROB):
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()
            augmented = self.color_augmentor(
                {'image': image_t, 'mask': mask_t})
            image = (augmented['image'] * 255.0).permute(
                1, 2, 0).numpy().astype(np.float32)
            mask = augmented['mask'][0].numpy().astype(np.uint8)
        return image, mask

    def _crop_eye(self, image, mask, cam_K, bbox_xyxy, rot_index,
                  im_H, im_W, dzi_rnd=None):
        """DZI crop one eye; mirrors ``BOP_Dataset.read_data``'s crop block.

        ``dzi_rnd=None`` draws the DZI zoom/pan per eye (mono path); stereo
        passes one shared draw so both eyes keep the eval-time geometric
        correspondence (each eye applies it relative to its own bbox).
        """
        if self.split in ('train', 'finetune'):
            bbox_center, bbox_scale, bbox_loc = misc.aug_bbox_DZI(
                bbox_xyxy, im_H, im_W,
                scale_ratio=self.DZI_SCALE_RATIO,
                shift_ratio=self.DZI_SHIFT_RATIO,
                pad_scale=self.DZI_PAD_SCALE, rnd=dzi_rnd)
        else:
            x1, y1, x2, y2 = bbox_xyxy.copy()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bbox_center = np.array([cx, cy])
            bbox_scale = self.DZI_PAD_SCALE * max(y2 - y1, x2 - x1)
            hr = (y2 - y1) / bbox_scale
            wr = (x2 - x1) / bbox_scale
            bbox_loc = np.array(
                [0.5 - wr/2, 0.5 - hr/2, 0.5 + wr/2, 0.5 + hr/2])

        rot_rad = rot_index * np.pi / 2
        Rz = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                       [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

        roi_mask = misc.crop_resize_by_warp_affine(
            mask, bbox_center, bbox_scale, self.mask_size, cam_K, rot_rad,
            interpolation='bilinear').squeeze(0)
        roi_mask = (roi_mask > 0.5).float()

        roi_img = misc.crop_resize_by_warp_affine(
            image, bbox_center, bbox_scale, self.rgb_size, cam_K, rot_rad,
            interpolation='bilinear') / 255.0

        T_rot = cam_K @ Rz @ np.linalg.inv(cam_K)
        center_hom = T_rot @ np.array(
            [*bbox_center, 1.0], dtype=np.float32)
        bbox_center = center_hom[:2]

        fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
        fov = np.array([(bbox_center[0] - cx) / fx,
                        (bbox_center[1] - cy) / fy, bbox_scale / fx])

        T_img2roi = misc.transform_to_local_ROIcrop(
            bbox_center=bbox_center, bbox_scale=bbox_scale,
            zoom_scale=self.rgb_size)
        roi_camK = T_img2roi.numpy() @ cam_K

        Tz = np.array([[1.0, 0.0, -0.5],
                       [0.0, 1.0, -0.5],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        T_loc = np.linalg.inv(Tz) @ Rz @ Tz
        bbox_loc = utils.transform_bounding_box(bbox_loc, T_loc)
        bbox_map = utils.make_roi(
            torch.as_tensor(bbox_loc, dtype=torch.float32),
            self.rgb_size).unsqueeze(0)

        return {
            'roi_image': torch.as_tensor(
                roi_img, dtype=torch.float32).contiguous(),
            'bbox_map': torch.as_tensor(bbox_map, dtype=torch.float32),
            'roi_camK': torch.as_tensor(
                roi_camK, dtype=torch.float32).squeeze(),
            'fov': torch.as_tensor(fov, dtype=torch.float32),
            'roi_mask': torch.as_tensor(
                roi_mask, dtype=torch.float32).contiguous(),
        }

    def read_data_stereo(self, dataset_dict, idx):
        """Left-eye targets + right-eye inputs for one matched instance.

        Both eyes share one Rz crop index and one DZI draw (the same zoom
        multiplier and relative pan applied to each eye's own bbox, so the
        crop pair keeps the geometric correspondence eval builds from the
        per-eye GT boxes); photometric augmentation stays per-eye
        independent.
        """
        info = dataset_dict['inst_infos']

        image_l = file_io.imread(
            info['image_file'], 'color', self.img_format).astype(np.float32)
        image_r = file_io.imread(
            info['image_file_right'], 'color',
            self.img_format).astype(np.float32)
        im_H, im_W = image_l.shape[:2]

        if cfg.CACHE_MASK:
            mask_l = misc.cocosegm2mask(info['mask_file'], im_H, im_W)
            mask_r = misc.cocosegm2mask(info['mask_file_right'], im_H, im_W)
        else:
            mask_l = file_io.imread(
                info['mask_file'], "unchanged").astype(bool).astype(np.uint8)
            mask_r = file_io.imread(
                info['mask_file_right'],
                "unchanged").astype(bool).astype(np.uint8)

        image_l, mask_l = self._maybe_color_aug(image_l, mask_l)
        image_r, mask_r = self._maybe_color_aug(image_r, mask_r)

        # One shared Rz crop index so both eyes rotate consistently.
        rot_index = (np.random.randint(4)
                     if self.split in ('train', 'finetune') else 0)
        dzi_rnd = misc.draw_dzi_randoms(
            self.DZI_SCALE_RATIO, self.DZI_SHIFT_RATIO)

        bbox_l = np.asarray(info['bbox'], dtype=np.float32)
        bbox_l = np.array([bbox_l[0], bbox_l[1],
                           bbox_l[0] + bbox_l[2], bbox_l[1] + bbox_l[3]])
        bbox_r = np.asarray(info['bbox_right'], dtype=np.float32)
        bbox_r = np.array([bbox_r[0], bbox_r[1],
                           bbox_r[0] + bbox_r[2], bbox_r[1] + bbox_r[3]])

        left = self._crop_eye(
            image_l, mask_l, info['intrinsics'], bbox_l, rot_index,
            im_H, im_W, dzi_rnd)
        right = self._crop_eye(
            image_r, mask_r, info['intrinsics_right'], bbox_r, rot_index,
            im_H, im_W, dzi_rnd)

        R_l = np.asarray(info['rotation'], dtype=np.float32).reshape(3, 3)
        t_l = np.asarray(info['translation'], dtype=np.float32).reshape(3)
        R_r = np.asarray(
            info['rotation_right'], dtype=np.float32).reshape(3, 3)
        t_r = np.asarray(
            info['translation_right'], dtype=np.float32).reshape(3)

        rot_rad = rot_index * np.pi / 2
        Rz = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                       [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        # Left -> right camera rigid transform, expressed in the Rz-rotated
        # crop frame the model composes the right render in.  With R_l == R_r
        # this reduces to [I | -Rz @ (b, 0, 0)], the stereobj-1m contract.
        # The rotations are orthonormalized first: the stored cam_R_m2c is only
        # orthonormal to ~2e-4, which would otherwise leak into T_R.
        R_rel = _orthonormalize(R_r) @ _orthonormalize(R_l).T
        T_right_ref = np.eye(4, dtype=np.float32)
        T_right_ref[:3, :3] = Rz @ R_rel @ Rz.T
        T_right_ref[:3, 3] = Rz @ (t_r - R_rel @ t_l)

        obj_id = info['obj_id']
        quat_bin = np.load(self.quat_label_path, mmap_mode='r')

        return {
            'roi_image': left['roi_image'],
            'bbox_map': left['bbox_map'],
            'roi_camK': left['roi_camK'],
            'fov': left['fov'],
            'obj_cls': torch.as_tensor(
                self.dataset_id2cls[obj_id], dtype=torch.int64),
            'roi_obj_R': torch.as_tensor(Rz @ R_l, dtype=torch.float32),
            'roi_obj_t': torch.as_tensor(Rz @ t_l, dtype=torch.float32),
            'roi_mask': left['roi_mask'],
            'quat_bin': torch.as_tensor(
                np.array(quat_bin[idx][rot_index], dtype=np.float32)),
            'obj_id': int(obj_id),
            'scene_id': int(info['scene_id']),
            'image_id': int(info['im_id']),
            'roi_image_right': right['roi_image'],
            'bbox_map_right': right['bbox_map'],
            'roi_camK_right': right['roi_camK'],
            'fov_right': right['fov'],
            'T_right_ref': torch.as_tensor(
                T_right_ref, dtype=torch.float32),
        }
