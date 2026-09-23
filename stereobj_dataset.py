"""Stereobj-1M webdataset adapter (``stereobj-1m`` dataset).

Source layout (under ``cfg.DATASET_ROOT/stereobj-1m``):

    webds/{train,val,test}-NNNNNN.tar   WebDataset shards
    objects/<name>.obj/.xyz/.bbox/.diameter/.kp
    split/<group>_{train,val,test}_scenes.txt, *_object_list.txt

Each webdataset sample (keyed ``<scene>_<frame>``) holds:

    *.webp      2000x1000 BGR side-by-side stereo pair; the left eye is
                x in [0, 1000), the right eye x in [1000, 2000). Both eyes
                are 1000x1000 after the converter downscaled the original
                two 1440x1440 squares.
    *.mask.png  2000x1000 uint8 instance mask; pixel value v covers
                ``Object {v-1}`` in both eyes.
    *.json      labels: ``class`` (``Object N`` -> model name), ``rt``
                (``Object N`` -> {R, t} in the *left* camera frame,
                metres), ``scaled_bboxes`` (per-eye boxes in full-image
                coords, from the converter).

Camera model (``camera.json``): a 3x4 projection P per eye for the
original 2880x1440 canvas (two 1440x1440 squares). Rescaling to
2000x1000 gives per-eye K with fx' = fx * 2000/2880,
cy' = cy * 1000/1440, cx'_left = cx * 2000/2880 and
cx'_right = cx * 2000/2880 + 1000 (the right eye is shifted, not
cropped, in the converted image). The right-eye projection carries the
stereo baseline in P[..., 3]; converted to metres via fx.

Modes:
    mono   each eye is an independent scene; the same PoseSample contract
           as ``BOP_Dataset`` (single view, left-frame pose). The right
           eye composes its own pose R, t - [b, 0, 0] in its own camera
           frame.
    stereo reserved for the two-view model: ``__getitem__`` returns both
           eyes' inputs plus the per-eye poses (see ``PoseSample`` note
           in dataset_schema); the training loop for this mode is not
           implemented yet.
"""
import io
import os
import tarfile

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

import config as cfg
import utils
from dataset_schema import PoseDataset
from object_registry import ObjectRegistry, SYMMETRY_EPS
from lib import data_utils as misc

import logging
logger = logging.getLogger(__name__)

cv2.setNumThreads(0)

SCHEMA_VERSION = 'stereobj-v1'

# camera.json projection matrix was defined on this canvas; the webds
# converter resized it to (WEBDS_W, WEBDS_H).
ORIG_W, ORIG_H = 2880, 1440
WEBDS_W, WEBDS_H = 2000, 1000
EYE_W = WEBDS_W // 2

# Fallback camera geometry in case camera.json is missing: fx was chosen
# so that the 63.34 px baseline offset (P[0, 3] / fx) matches the known
# ~45 mm baseline of the rig.
DEFAULT_FX_ORIG = 1408.8347397136001
DEFAULT_CX_ORIG = 711.4153137207031
DEFAULT_CY_ORIG = 787.1127624511719
BASELINE_PX_ORIG = 63.34150140067935

EYE_KEYS = ['left', 'right']


class StereobjDataset(PoseDataset):
    """Random-access loader for the stereobj-1m webdataset."""

    def __init__(self, dataset_name='stereobj-1m', split='train', rank=0):
        self.dataset_name = dataset_name
        self.rgb_size = cfg.INPUT_IMG_SIZE
        self.mask_size = cfg.OUTPUT_MASK_SIZE
        self.data_dir = os.path.join(cfg.DATASET_ROOT, dataset_name)
        ds_cfg = cfg.DATASET_CONFIG[dataset_name]
        self.ds_cfg = ds_cfg
        self.split = split
        self.stereo_mode = ds_cfg.get('stereo_mode', 'mono')

        self.width = ds_cfg['width']
        self.height = ds_cfg['height']
        self.depth_min = ds_cfg['Tz_near']
        self.depth_max = ds_cfg['Tz_far']
        self.num_objects = ds_cfg['num_class']
        self.dataset_id2cls = ds_cfg['id2cls']

        set_key = {'train': 'train_set', 'finetune': 'finetune_set',
                   'val': 'val_set', 'test': 'test_set'}[split]
        self.name_set = ds_cfg[set_key]
        assert isinstance(self.name_set, list), \
            'train_set(s) must be a list'

        self.img_format = 'BGR'
        self.mask_morph = True
        self.mask_morph_kernel_size = 3

        self.DZI_PAD_SCALE = cfg.ZOOM_PAD_SCALE
        self.DZI_SCALE_RATIO = cfg.ZOOM_SCALE_RATIO
        self.DZI_SHIFT_RATIO = cfg.ZOOM_SHIFT_RATIO
        self.Rz_rotation_aug = cfg.RZ_ROTATION_AUG
        self.COLOR_AUG_PROB = cfg.COLOR_AUG_PROB
        self.CHANGE_BG_PROB = cfg.CHANGE_BG_PROB  # unused: no bg corpus yet

        # Photometric augmentation, identical to BOP_Dataset's.
        self.color_augmentor = v2.Compose([
            v2.RandomApply([v2.GaussianNoise(sigma=0.01)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.5,
                                           saturation=0.5, hue=0.05)], p=0.5),
            v2.RandomInvert(p=0.15),
            v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.5, 2.0), value=0.0),
        ])

        self._camera = self._load_camera()
        self.registry = ObjectRegistry(dataset_name)

        # Frame-level index over the shards (member offsets + labels),
        # cached as a pickle; expanded to a columnar (frame, instance,
        # eye) sample table in-memory (see _expand_index).
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        # name -> class index, inverted once for the columnar table.
        self.cls_names = [None] * self.num_objects
        for name, cls in self.dataset_id2cls.items():
            self.cls_names[cls] = name
        webds_dir = os.path.join(self.data_dir, self.ds_cfg['webds_dir'])
        self._shard_paths = sorted(
            os.path.join(webds_dir, f) for f in os.listdir(webds_dir)
            if any(f.startswith(s + '-') for s in self.name_set)
            and f.endswith('.tar'))
        self.frame_records = self._build_index()
        self._expand_index()

    def _expand_index(self):
        """Columnar flat sample table over (frame, instance, eye).

        Replaces the nested per-frame dict lists with plain numpy arrays:
        with 2.2M mono samples the dict form costs ~380 MB *per worker*
        after Windows spawn, which thrashes the box. Columnar storage is
        ~15x smaller and each worker holds a COW-shared copy.

        mono:   one sample per (instance, annotated eye); each eye is an
                independent scene.
        stereo: one sample per instance annotated on both eyes; reserved
                for the two-view model (raises in __getitem__).
        """
        n_frames = len(self.frame_records)

        # Columnar frame table.
        self._f_shard = np.array([
            self._shard_paths.index(f['shard']) for f in self.frame_records],
            dtype=np.int32)
        self._f_json_offset = np.array(
            [f['json_offset'] for f in self.frame_records], dtype=np.int64)
        self._f_json_size = np.array(
            [f['json_size'] for f in self.frame_records], dtype=np.int64)
        self._f_mask_offset = np.array(
            [f['mask_offset'] for f in self.frame_records], dtype=np.int64)
        self._f_mask_size = np.array(
            [f['mask_size'] for f in self.frame_records], dtype=np.int64)
        self._f_webp_offset = np.array(
            [f['webp_offset'] for f in self.frame_records], dtype=np.int64)
        self._f_webp_size = np.array(
            [f['webp_size'] for f in self.frame_records], dtype=np.int64)
        self._f_scene_id = np.array(
            [f['scene_id'] for f in self.frame_records], dtype=np.int32)
        self._f_image_id = np.array(
            [int(f['key'].rsplit('_', 1)[1]) for f in self.frame_records],
            dtype=np.int32)

        # Columnar instance table, frame-major (row block per frame).
        # A mono sample index maps to (instance row, eye) via _offsets.
        ii_frame = []   # frame index per instance
        ii_mask = []    # mask id per instance
        ii_sides = []   # left/right annotation bits per instance
        ii_cls = []     # class index per instance
        ii_R = []       # (N, 3, 3) rotation in the left-cam frame
        ii_t = []       # (N, 3) translation in the left-cam frame
        for fi, frame in enumerate(self.frame_records):
            for inst in frame['instances']:
                ii_frame.append(fi)
                ii_mask.append(inst['mask_id'])
                ii_sides.append(inst['sides'])
                ii_cls.append(self.dataset_id2cls[inst['obj_name']])
                ii_R.append(inst['R'])
                ii_t.append(inst['t'])
        self._i_frame = np.array(ii_frame, dtype=np.int32)
        self._i_mask = np.array(ii_mask, dtype=np.int32)
        self._i_sides = np.array(ii_sides, dtype=np.uint8)
        self._i_cls = np.array(ii_cls, dtype=np.int32)
        self._i_R = np.stack(ii_R).astype(np.float32)
        self._i_t = np.stack(ii_t).astype(np.float32)

        # Per (instance, eye) -> packed row (instance_row << 1 | eye) of
        # the eye-expansion, grouped frame-major so a flat sample index
        # resolves via searchsorted on _offsets + _frame_row_start.
        if self.stereo_mode == 'stereo':
            rows_all = np.where(self._i_sides == 0b11)[0] << 1
        else:
            lefts = np.where(self._i_sides & 1)[0] << 1
            rights = (np.where(self._i_sides & 2)[0] << 1) | 1
            # eye-major within instance: (inst0-left, inst0-right, ...)
            keys = np.concatenate([
                np.stack([lefts >> 1, lefts], axis=1),
                np.stack([rights >> 1, rights], axis=1)])
            order = np.lexsort((keys[:, 1], keys[:, 0]))
            rows_all = keys[order, 1]
        side_any = (self._i_sides == 0b11) if self.stereo_mode == 'stereo' \
            else ((self._i_sides & 0b11) > 0)
        inst_rows = np.where(side_any)[0]
        inst_frames = self._i_frame[inst_rows]
        if self.stereo_mode == 'stereo':
            samples_per_inst = np.ones(len(inst_rows), dtype=np.int64)
        else:
            samples_per_inst = np.array(
                [bin(s).count('1') for s in self._i_sides[inst_rows]],
                dtype=np.int64)
        # Count samples per frame, then scatter rows into frame-major order.
        per_frame = np.bincount(
            inst_frames, weights=samples_per_inst, minlength=n_frames)
        self._offsets = np.concatenate(
            [[0], np.cumsum(per_frame)]).astype(np.int64)
        self._n_samples = int(self._offsets[-1])
        self._frame_row_start = self._offsets.copy()
        self._frame_rows = np.asarray(rows_all, dtype=np.int64)

        # Release the dict-form records; workers only need the columns.
        self.frame_records = None

    # ------------------------------------------------------------------ #
    # PoseDataset contract
    # ------------------------------------------------------------------ #
    def get_info(self):
        return {
            'depth_min': self.depth_min,
            'depth_max': self.depth_max,
            'num_objects': self.num_objects}

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx):
        if self.stereo_mode != 'mono':
            raise NotImplementedError(
                "stereo mode is scaffolded but the two-view model is not "
                "implemented; use stereo_mode='mono'")
        fi = int(np.searchsorted(self._offsets, idx, side='right') - 1)
        off = idx - self._offsets[fi]
        packed = int(self._frame_rows[self._frame_row_start[fi] + off])
        row, eye_i = packed >> 1, packed & 1
        return self.read_data(fi, row, EYE_KEYS[eye_i])

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #
    def _load_camera(self):
        """Per-eye 3x3 intrinsics on the webds canvas + baseline (m)."""
        import json
        cam_path = os.path.join(self.data_dir, 'camera.json')
        if os.path.exists(cam_path):
            with open(cam_path, 'r') as fp:
                cam = json.load(fp)
        else:
            cam = {
                'left': {'P': [[DEFAULT_FX_ORIG, 0, DEFAULT_CX_ORIG, 0],
                               [0, DEFAULT_FX_ORIG, DEFAULT_CY_ORIG, 0],
                               [0, 0, 1, 0]]},
                'right': {'P': [[DEFAULT_FX_ORIG, 0, DEFAULT_CX_ORIG,
                                 -BASELINE_PX_ORIG],
                                [0, DEFAULT_FX_ORIG, DEFAULT_CY_ORIG, 0],
                                [0, 0, 1, 0]]},
            }
        sx, sy = WEBDS_W / ORIG_W, WEBDS_H / ORIG_H
        out = {}
        for side, entry in cam.items():
            P = np.asarray(entry['P'], dtype=np.float64)
            fx = P[0, 0] * sx
            fy = P[1, 1] * sy
            cx = P[0, 2] * sx + (EYE_W if side == 'right' else 0.0)
            cy = P[1, 2] * sy
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
            # P_right = [K | -fx*b]: baseline b = |P[0,3]| / fx in metres,
            # independent of the pixel scale.
            baseline = abs(P[0, 3]) / P[0, 0]
            out[side] = {'K': K, 'baseline_m': float(baseline)}
        return out

    @property
    def baseline(self):
        """Stereo baseline in metres (right-cam pose offset)."""
        return self._camera['right']['baseline_m']

    def eye_intrinsics(self, eye):
        return self._camera[eye]['K']

    # ------------------------------------------------------------------ #
    # Index: tar member offsets, cached
    # ------------------------------------------------------------------ #
    def _index_cache_path(self):
        import hashlib
        digest = hashlib.md5("_".join(
            [SCHEMA_VERSION, self.dataset_name, self.data_dir,
             "_".join(self.name_set), self.stereo_mode]
        ).encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir,
                            "stereobj_index_{}_{}.pkl".format(
                                "_".join(self.name_set), digest))

    def _build_index(self):
        """Scan shards once: member offsets + parsed labels per frame.

        Produces a compact frame-level record list:
            key, scene, scene_id, shard, member offsets/sizes, and a list
            of instances {obj_key, obj_name, mask_id, R, t, sides}, where
            ``sides`` bit 0/1 marks left/right eye annotation presence.
        """
        from lib import file_io
        cache_path = self._index_cache_path()
        if cfg.USE_CACHE and os.path.exists(cache_path):
            frames = file_io.load(cache_path)
            n_inst = sum(len(f['instances']) for f in frames)
            logger.info("loaded stereobj index: %d frames / %d instances",
                        len(frames), n_inst)
            return frames

        shard_paths = self._shard_paths
        assert shard_paths, 'no webds shards for set(s) {}'.format(self.name_set)

        frames = []
        for shard_path in shard_paths:
            with tarfile.open(shard_path, 'r') as tar:
                # Stream members; each sample holds three members in
                # shard order: <key>.json, <key>.mask.png, <key>.webp.
                pending = {}
                for member in tar:
                    name = member.name
                    if name.endswith('/'):
                        continue
                    stem, ext = name.rsplit('.', 1)[0], name.rsplit('.', 1)[-1]
                    if ext == 'json':
                        pending['json_offset'] = member.offset_data
                        pending['json_size'] = member.size
                    elif ext == 'png':
                        pending['mask_offset'] = member.offset_data
                        pending['mask_size'] = member.size
                    elif ext == 'webp':
                        pending['webp_offset'] = member.offset_data
                        pending['webp_size'] = member.size
                    if len(pending) == 6:
                        frame = self._parse_sample(
                            shard_path, stem, tar, pending)
                        if frame is not None:
                            frames.append(frame)
                        pending = {}

        scenes = sorted({f['scene'] for f in frames})
        scene_id_map = {s: i for i, s in enumerate(scenes)}
        for f in frames:
            f['scene_id'] = scene_id_map[f['scene']]

        if cfg.USE_CACHE:
            file_io.dump(frames, cache_path)
            logger.info("dumped stereobj index to %s", cache_path)
        n_inst = sum(len(f['instances']) for f in frames)
        logger.info("stereobj index: %d frames / %d instances / %d scenes",
                    len(frames), n_inst, len(scenes))
        return frames

    def _parse_sample(self, shard_path, stem, tar, pending):
        import json
        label = json.loads(self._read_member(
            tar, pending['json_offset'], pending['json_size']).decode('utf-8'))
        classes = label.get('class', {})
        rt = label.get('rt', {})
        boxes = label.get('scaled_bboxes', {})
        instances = []
        for obj_key, obj_name in classes.items():
            if obj_name not in self.dataset_id2cls or obj_key not in rt:
                continue
            sides = 0
            for eye_i, eye in enumerate(EYE_KEYS):
                if boxes.get(eye, {}).get(obj_key) is not None:
                    sides |= 1 << eye_i
            if sides == 0:
                continue
            instances.append({
                'obj_key': obj_key,
                'obj_name': obj_name,
                'mask_id': int(obj_key.split()[-1]) + 1,
                'R': np.asarray(rt[obj_key]['R'], dtype=np.float32).reshape(3, 3),
                't': np.asarray(rt[obj_key]['t'], dtype=np.float32).reshape(3),
                'sides': sides,
            })
        if not instances:
            return None
        return {
            'shard': shard_path,
            'key': stem,
            'scene': stem.rsplit('_', 1)[0],
            'instances': instances,
            **pending,
        }

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _read_member(self, tar, offset, size):
        tar.fileobj.seek(offset)
        return tar.fileobj.read(size)

    def read_data(self, fi, row, eye):
        """Load one (frame, instance, eye) and build a PoseSample dict."""
        import json
        x_off = EYE_W if eye == 'right' else 0
        shard = self._shard_paths[self._f_shard[fi]]

        with tarfile.open(shard, 'r') as tar:
            label = json.loads(self._read_member(
                tar, self._f_json_offset[fi],
                self._f_json_size[fi]).decode('utf-8'))
            webp_buf = self._read_member(
                tar, self._f_webp_offset[fi], self._f_webp_size[fi])
            mask_buf = self._read_member(
                tar, self._f_mask_offset[fi], self._f_mask_size[fi])

        img_pair = cv2.imdecode(
            np.frombuffer(webp_buf, np.uint8), cv2.IMREAD_COLOR)  # BGR
        image = np.ascontiguousarray(img_pair[:, x_off:x_off + EYE_W])
        im_H, im_W = image.shape[:2]

        mask_full = cv2.imdecode(
            np.frombuffer(mask_buf, np.uint8), cv2.IMREAD_UNCHANGED)
        mask_full = np.ascontiguousarray(mask_full[:, x_off:x_off + EYE_W])

        cam_K = self.eye_intrinsics(eye)

        # Pose: json rt is in the left-camera frame. The right camera sits
        # at +baseline along x, so its pose composes as t_r = t_l - [b,0,0]
        # (mirrored by P_right = [K | -fx*b] in camera.json).
        R = self._i_R[row].copy()
        t = self._i_t[row].copy()
        if eye == 'right':
            t = t - np.array([self.baseline, 0.0, 0.0], dtype=np.float32)

        mask = (mask_full == self._i_mask[row]).astype(np.uint8)
        if self.mask_morph:
            kernel = np.ones((self.mask_morph_kernel_size,
                              self.mask_morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        obj_key = 'Object {}'.format(self._i_mask[row] - 1)
        obj_name = self.cls_names[self._i_cls[row]]
        bbox_entry = label['scaled_bboxes'][eye][obj_key]
        x1 = bbox_entry['x_min'] - x_off
        x2 = bbox_entry['x_max'] - x_off
        y1, y2 = bbox_entry['y_min'], bbox_entry['y_max']
        bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Photometric augmentation (same path as BOP_Dataset).
        if (self.split in ('train', 'finetune')
                and np.random.rand() < self.COLOR_AUG_PROB):
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()
            augmented = self.color_augmentor({'image': image_t, 'mask': mask_t})
            image = (augmented['image'] * 255.0).permute(1, 2, 0).numpy().astype(np.float32)
            mask = augmented['mask'][0].numpy().astype(np.uint8)

        # DZI crop (same as BOP_Dataset).
        if self.split in ('train', 'finetune'):
            bbox_center, bbox_scale, bbox_loc = misc.aug_bbox_DZI(
                bbox_xyxy, im_H, im_W,
                scale_ratio=self.DZI_SCALE_RATIO,
                shift_ratio=self.DZI_SHIFT_RATIO,
                pad_scale=self.DZI_PAD_SCALE)
        else:
            bx1, by1, bx2, by2 = bbox_xyxy
            cx = 0.5 * (bx1 + bx2)
            cy = 0.5 * (by1 + by2)
            bbox_center = np.array([cx, cy])
            bbox_scale = self.DZI_PAD_SCALE * max(by2 - by1, bx2 - bx1)
            hr = (by2 - by1) / bbox_scale
            wr = (bx2 - bx1) / bbox_scale
            bbox_loc = np.array([0.5 - wr/2, 0.5 - hr/2,
                                 0.5 + wr/2, 0.5 + hr/2])

        rot_index = (np.random.randint(4)
                     if self.split in ('train', 'finetune') else 0)
        rot_rad = rot_index * np.pi / 2
        Rz = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                       [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        R = Rz @ R
        t = Rz @ t

        roi_mask = misc.crop_resize_by_warp_affine(
            mask, bbox_center, bbox_scale, self.mask_size, cam_K, rot_rad,
            interpolation='bilinear').squeeze(0)
        roi_mask = (roi_mask > 0.5).float()

        roi_img = misc.crop_resize_by_warp_affine(
            image, bbox_center, bbox_scale, self.rgb_size, cam_K, rot_rad,
            interpolation='bilinear') / 255.0

        T_rot = cam_K @ Rz @ np.linalg.inv(cam_K)
        center_hom = T_rot @ np.array([*bbox_center, 1.0], dtype=np.float32)
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

        quat_bin = self._compute_quat_bin(obj_name, R, rot_index)

        return {
            'roi_image': torch.as_tensor(roi_img, dtype=torch.float32).contiguous(),
            'bbox_map': torch.as_tensor(bbox_map, dtype=torch.float32),
            'roi_camK': torch.as_tensor(roi_camK, dtype=torch.float32).squeeze(),
            'fov': torch.as_tensor(fov, dtype=torch.float32),
            'obj_cls': torch.as_tensor(int(self._i_cls[row]), dtype=torch.int64),
            'roi_obj_R': torch.as_tensor(R, dtype=torch.float32),
            'roi_obj_t': torch.as_tensor(t, dtype=torch.float32),
            'roi_mask': torch.as_tensor(roi_mask, dtype=torch.float32).contiguous(),
            'quat_bin': torch.as_tensor(quat_bin, dtype=torch.float32),
            # numeric object id (class index + 1), mirroring BOP ids
            'obj_id': int(self._i_cls[row]) + 1,
            'scene_id': int(self._f_scene_id[fi]) * 2 + EYE_KEYS.index(eye),
            'image_id': int(self._f_image_id[fi]),
        }

    # ------------------------------------------------------------------ #
    # Quaternion labels (lazy; stereobj-1m is far too large to precompute)
    # ------------------------------------------------------------------ #
    def _compute_quat_bin(self, obj_name, obj_R, rot_index):
        """(N_POSE_BIN,) label for the given Rz-composed rotation.

        Mirrors the per-instance rows ``BOP_Dataset`` precomputes, but
        stereobj-1m has far too many instances for an (N, 4, N_POSE_BIN)
        table, so the bin distribution is quantized on demand (CPU, ~0.2
        s/sample, parallelized by the DataLoader workers).
        """
        cls = self.dataset_id2cls[obj_name]
        registry = self.registry
        num_v = int(registry.quant_points_mask[cls].sum().item())
        verts = registry.quant_points[cls][:num_v]
        vmask = registry.quant_points_mask[cls][:num_v]
        diam = registry.diameter[cls]
        vcorr = registry.quant_correlation[cls]
        qsym = registry.quaternion_symmetries[cls]
        tsym = registry.translation_symmetries[cls]
        smask = registry.symmetries_mask[cls]

        rz = np.stack([[[np.cos(r * np.pi / 2), -np.sin(r * np.pi / 2), 0.],
                        [np.sin(r * np.pi / 2), np.cos(r * np.pi / 2), 0.],
                        [0., 0., 1.]] for r in range(4)]).astype(np.float32)
        composed = np.einsum('rij,jk->rik', rz, obj_R)
        quats = utils.rotation_to_quaternion(
            torch.from_numpy(composed))

        # Stay on CPU: lazy labels run inside DataLoader workers, and CUDA
        # contexts per worker are too heavy (Windows spawn, VRAM sharing).
        cats, _, _ = utils.quantize_quaternion_vertex(
            quats,
            verts[None].expand(4, -1, -1).contiguous(),
            vmask[None].expand(4, -1).contiguous(),
            diam[None].expand(4).contiguous(),
            vcorr[None].expand(4, -1, -1).contiguous(),
            qsym[None].expand(4, -1, -1).contiguous(),
            tsym[None].expand(4, -1, -1).contiguous(),
            smask[None].expand(4, -1).contiguous())
        return cats[rot_index].numpy()
