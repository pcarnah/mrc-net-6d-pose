import os
import sys
from pathlib import Path
import cv2
import torch
import random
import hashlib
import numpy as np
from tqdm import tqdm
import os.path as osp
import utils
import config as cfg
from dataset_schema import PoseDataset
from object_registry import ObjectRegistry, SYMMETRY_EPS

import logging
logger = logging.getLogger(__name__)

CUR_FILE_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(CUR_FILE_DIR, '..'))
sys.path.append(PROJ_ROOT)
cv2.setNumThreads(0)

SCHEMA_VERSION = 'v2'

# Quaternion-bin label generation is chunked on GPU; one sample of the
# inner computation materializes (N, N_POSE_BIN, V, 3). Budget in bytes and
# a hard cap keep peak memory bounded across datasets/objects.
QUAT_LABEL_MEM_BUDGET = 3_000_000_000
QUAT_LABEL_CHUNK_MAX = 16
QUAT_LABEL_GATE_INSTANCES = 200

from lib import data_utils as misc
from lib import file_io
from torchvision.transforms import v2


class BOP_Dataset(PoseDataset):
    def __init__(self, dataset_name, split, rank=0):
        self.dataset_name = dataset_name
        self.rgb_size = cfg.INPUT_IMG_SIZE
        self.mask_size = cfg.OUTPUT_MASK_SIZE
        self.data_dir = os.path.join(cfg.DATASET_ROOT, dataset_name)

        self.width = cfg.DATASET_CONFIG[dataset_name]['width']
        self.height = cfg.DATASET_CONFIG[dataset_name]['height']
        self.color_type = cfg.DATASET_CONFIG[dataset_name]['color_type']
        self.split = split

        self.depth_min = cfg.DATASET_CONFIG[dataset_name]['Tz_near']
        self.depth_max = cfg.DATASET_CONFIG[dataset_name]['Tz_far']
        self.num_objects = cfg.DATASET_CONFIG[dataset_name]['num_class']

        if split == 'train':
            self.name_set = cfg.DATASET_CONFIG[dataset_name]['train_set']
        elif split == 'finetune':
            self.name_set = cfg.DATASET_CONFIG[dataset_name]['finetune_set']
        else:
            self.name_set = cfg.DATASET_CONFIG[dataset_name]['test_set']

        assert(isinstance(self.name_set, list)), 'train_set(s) must be a list' # ['train_pbr', 'train_real', ...]
        self.dataset_id2cls = cfg.DATASET_CONFIG[dataset_name]['id2cls']
        self.num_classes = len(self.dataset_id2cls)

        self.img_format = 'BGR'
        self.mask_morph = True
        self.filter_invalid = True
        self.mask_morph_kernel_size = 3
        # Photometric augmentation, applied to the crop together with its mask.
        # The image is fed as a float tensor in [0, 1]; the mask is passed
        # through untouched by the colour transforms and only erased alongside
        # the dropped-out regions.
        self.color_augmentor = v2.Compose([
            v2.RandomApply([v2.GaussianNoise(sigma=0.01)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.5,
                                           saturation=0.5, hue=0.05)], p=0.5),
            v2.RandomInvert(p=0.15),
            v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.5, 2.0), value=0.0),
        ])

        self.DZI_PAD_SCALE = cfg.ZOOM_PAD_SCALE
        self.DZI_SCALE_RATIO = cfg.ZOOM_SCALE_RATIO  # wh scale
        self.DZI_SHIFT_RATIO = cfg.ZOOM_SHIFT_RATIO  # center shift
        self.Rz_rotation_aug = cfg.RZ_ROTATION_AUG
        self.CHANGE_BG_PROB = cfg.CHANGE_BG_PROB
        self.COLOR_AUG_PROB = cfg.COLOR_AUG_PROB

        self.TRUNCATE_FG = False
        self.BG_KEEP_ASPECT_RATIO = True
        self.NUM_BG_IMGS = 10000
        self.BG_TYPE = "VOC_table"      # VOC_table | coco | VOC | SUN2012
        self.BG_ROOT = cfg.VOC_BG_ROOT  # "datasets/coco/train2017/"

        self.use_cache = cfg.USE_CACHE
        self.cache_dir = os.path.join(CUR_FILE_DIR, ".cache")  # .cache

        hashed_file_name = hashlib.md5(("_".join(self.name_set)
            + "dataset_dicts_{}_{}_{}_{}".format(
                SCHEMA_VERSION, self.dataset_name, self.data_dir, __name__)
        ).encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir,
            "dataset_dicts_{}_{}_{}.pkl".format(self.dataset_name, "_".join(self.name_set), hashed_file_name))

        self.model_folders = cfg.DATASET_CONFIG[dataset_name]['model_folders']

        self.dataset_dicts = list()
        if self.use_cache and os.path.exists(cache_path):
            # print("load cached dataset dicts from {}".format(cache_path))
            self.dataset_dicts = file_io.load(cache_path)
            # print('done')
        else:
            for img_type in self.name_set:
                image_counter = 0
                instance_counter = 0
                train_dir = os.path.join(self.data_dir, img_type)
                logger.info("preparing data from {}".format(img_type))
                model_folder = os.path.join(self.data_dir, self.model_folders[img_type])

                ## process scene and images ############
                for scene in sorted(os.listdir(train_dir)):  # scene
                    if not scene.startswith('00'):  # BOP images start with '0000xx'
                        continue
                    scene_id = int(scene)
                    scene_dir = os.path.join(train_dir, scene)
                    scene_cam_dict = file_io.load(os.path.join(scene_dir, "scene_camera.json"))      # gt_intrinsic
                    scene_gt_pose_dict = file_io.load(os.path.join(scene_dir, "scene_gt.json"))      # gt_poses
                    scene_gt_bbox_dict = file_io.load(os.path.join(scene_dir, "scene_gt_info.json"))  # gt_bboxes
                    for img_id_str in tqdm(scene_gt_pose_dict, postfix=f"{scene_id}"):  # image
                        img_id_int = int(img_id_str)
                        color_type = self.color_type
                        rgb_path = os.path.join(scene_dir, "{}/{:06d}.jpg").format(color_type, img_id_int)
                        if not os.path.exists(rgb_path):
                            rgb_path = os.path.join(scene_dir, "{}/{:06d}.png").format(color_type, img_id_int)
                        if not os.path.exists(rgb_path):
                            rgb_path = os.path.join(scene_dir, "{}/{:06d}.tif").format(color_type, img_id_int)
                        if not os.path.exists(rgb_path):
                            rgb_path = os.path.join(scene_dir, "{}/{:06d}.bmp").format(color_type, img_id_int)
                        assert os.path.exists(rgb_path), rgb_path
                        cam_K = np.array(scene_cam_dict[img_id_str]["cam_K"], dtype=np.float32).reshape(3, 3)

                        record = {
                            "dataset_name": self.dataset_name,
                            "scene_id": scene_id,
                            "image_id": img_id_int,
                            "img_type": img_type,
                            "height": self.height,
                            "width": self.width,
                        }
                        view_insts = []
                        view_inst_count = dict() # count the object number per instance in a single image
                        for anno_idx, anno_dict in enumerate(scene_gt_pose_dict[img_id_str]):
                            obj_id = anno_dict["obj_id"]
                            if obj_id not in self.dataset_id2cls: # ignore the non-target objects 
                                continue
                            R = np.array(anno_dict["cam_R_m2c"], dtype="float32").reshape(3, 3)
                            t = np.array(anno_dict["cam_t_m2c"], dtype="float32")

                            quat_path = os.path.join(scene_dir, "quat_label/{:06d}_{:06d}.npy").format(img_id_int, anno_idx)

                            bbox_visib = scene_gt_bbox_dict[img_id_str][anno_idx]["bbox_visib"]
                            x1, y1, w, h = bbox_visib
                            if self.filter_invalid:
                                if h <= 10 or w <= 10:
                                    continue
                            ### Load precompute quaternion bin id and residual #########
                            model_path = os.path.join(
                                model_folder, 'obj_{:06d}.ply'.format(int(obj_id)))

                            mask_visib_file = os.path.join(scene_dir, "mask_visib/{:06d}_{:06d}.png".format(img_id_int, anno_idx))
                            assert os.path.exists(mask_visib_file), mask_visib_file
                            visib_fract = scene_gt_bbox_dict[img_id_str][anno_idx]["visib_fract"]
                            if visib_fract < 0.10:  # filter out too small or nearly invisible instances
                                continue

                            if cfg.CACHE_MASK:
                                mask_single = file_io.imread(mask_visib_file, "unchanged").astype(bool).astype(np.uint8)
                                if self.mask_morph:
                                    kernel = np.ones((self.mask_morph_kernel_size, self.mask_morph_kernel_size))
                                    mask_single = cv2.morphologyEx(mask_single.astype(np.uint8), cv2.MORPH_CLOSE, kernel)  # remove holes
                                    mask_single = cv2.morphologyEx(mask_single, cv2.MORPH_OPEN, kernel)  # remove outliers
                                mask_single = misc.binary_mask_to_rle(mask_single, compressed=True)

                            else:
                                mask_single = mask_visib_file

                            if obj_id not in view_inst_count:
                                view_inst_count[obj_id] = 0
                            view_inst_count[obj_id] += 1  # accumulate the object number per instance in a single image

                            # Object instance level information dict
                            inst = {
                                'sub_dataset_folder': img_type,
                                'image_file': rgb_path,
                                'mask_file': mask_single,
                                'model_file': model_path,
                                'bbox': bbox_visib,
                                'quat_file': quat_path,
                                'rotation': R,
                                'translation': t,
                                'intrinsics': cam_K,
                                'scene_id': scene_id,
                                'im_id': img_id_int,
                                'obj_id': int(obj_id),
                            }

                            view_insts.append(inst)
                        if len(view_insts) == 0:  # filter im without anno
                            continue
                        record["annotations"] = view_insts
                        record['obj_inst_count'] = view_inst_count
                        self.dataset_dicts.append(record)

                        image_counter += 1
                        instance_counter += len(view_insts)

                    print(img_type, ', images: ', image_counter, ', instances: ', instance_counter)

                file_io.dump(self.dataset_dicts, cache_path, protocol=5)
                logger.info("Dumped dataset_dicts to {}".format(cache_path))

        self.dataset_dicts = misc.flat_dataset_dicts(self.dataset_dicts) # flatten the image-level dict to instance-level dict

        self.quat_label_path = self._quat_label_cache_path()
        if not os.path.exists(self.quat_label_path):
            labels = self._generate_quat_labels()
            self._verify_quat_labels(labels)
            np.save(self.quat_label_path, labels.astype(np.float16))
            logger.info("Saved quaternion labels to {}".format(
                self.quat_label_path))

    def _instance_fingerprint(self):
        """sha256 over the ordered (obj_id, R, t) of every instance."""
        h = hashlib.sha256()
        for inst in self.dataset_dicts:
            info = inst['inst_infos']
            h.update(str(int(info['obj_id'])).encode('utf-8'))
            h.update(np.ascontiguousarray(
                info['rotation'], dtype=np.float32).tobytes())
            h.update(np.ascontiguousarray(
                info['translation'], dtype=np.float32).tobytes())
        return h.hexdigest()

    def _quat_label_cache_path(self):
        digest = hashlib.sha256("_".join([
            SCHEMA_VERSION,
            str(cfg.N_POSE_BIN),
            str(cfg.POSE_SIGMA),
            "proto{}".format(utils.PROTOTYPE_VERSION),
            str(SYMMETRY_EPS),
            str(len(self.dataset_dicts)),
            self._instance_fingerprint(),
        ]).encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, "quatbin_{}_{}_{}.npy".format(
            self.dataset_name, "_".join(self.name_set), digest))

    def _quat_label_chunk(self, registry, cls):
        num_v = int(registry.num_vertices[cls])
        num_s = max(int(registry.symmetries_mask[cls].sum().item()), 1)
        per_sample = cfg.N_POSE_BIN * (num_v * 3 + num_s * 13) * 4
        chunk = int(QUAT_LABEL_MEM_BUDGET // max(per_sample, 1))
        return max(1, min(QUAT_LABEL_CHUNK_MAX, chunk))

    def _quantize_quat_batched(self, registry, cls, quaternions, device):
        num_v = int(registry.num_vertices[cls])
        verts = registry.vertices[cls][:num_v]
        vmask = registry.vertices_mask[cls][:num_v]
        diam = registry.diameter[cls]
        vcorr = registry.vertices_correlation[cls]
        qsym = registry.quaternion_symmetries[cls]
        tsym = registry.translation_symmetries[cls]
        smask = registry.symmetries_mask[cls]

        chunk = self._quat_label_chunk(registry, cls)
        categories = []
        for start in range(0, len(quaternions), chunk):
            quat = quaternions[start:start + chunk].to(device)
            n = len(quat)
            cat = utils.quantize_quaternion_vertex(
                quat,
                verts[None].expand(n, -1, -1).contiguous().to(device),
                vmask[None].expand(n, -1).contiguous().to(device),
                diam[None].expand(n).contiguous().to(device),
                vcorr[None].expand(n, -1, -1).contiguous().to(device),
                qsym[None].expand(n, -1, -1).contiguous().to(device),
                tsym[None].expand(n, -1, -1).contiguous().to(device),
                smask[None].expand(n, -1).contiguous().to(device))[0]
            categories.append(cat.detach().cpu().numpy())
        return np.concatenate(categories, axis=0)

    def _generate_quat_labels(self):
        """Compute (N_inst, 4, N_POSE_BIN) labels on GPU, grouped by object."""
        n_inst = len(self.dataset_dicts)
        n_bins = cfg.N_POSE_BIN
        out = np.empty((n_inst, 4, n_bins), dtype=np.float32)
        registry = ObjectRegistry(self.dataset_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        by_cls = {}
        for i, inst in enumerate(self.dataset_dicts):
            cls = self.dataset_id2cls[inst['inst_infos']['obj_id']]
            by_cls.setdefault(cls, []).append(i)

        for cls, idxs in tqdm(by_cls.items(), desc='quat labels'):
            rotations = np.stack([
                np.asarray(self.dataset_dicts[i]['inst_infos']['rotation'],
                           dtype=np.float32).reshape(3, 3) for i in idxs])
            rz = np.stack([[[np.cos(r * np.pi / 2), -np.sin(r * np.pi / 2), 0.],
                            [np.sin(r * np.pi / 2), np.cos(r * np.pi / 2), 0.],
                            [0., 0., 1.]] for r in range(4)])
            rz = rz.astype(np.float32)
            composed = np.einsum('rij,njk->nrik', rz, rotations)
            quats = utils.rotation_to_quaternion(
                torch.from_numpy(composed.reshape(-1, 3, 3)))
            labels = self._quantize_quat_batched(
                registry, cls, quats, device)
            out[idxs] = labels.reshape(len(idxs), 4, n_bins)
        return out

    def _verify_quat_labels(self, labels):
        """Equivalence gate against the legacy ``quat_label/*.npy`` sidecars."""
        legacy = [i for i in range(len(self.dataset_dicts))
                  if os.path.exists(
                      self.dataset_dicts[i]['inst_infos']['quat_file'])]
        if len(legacy) == 0:
            logger.warning(
                "No legacy quat_label/*.npy found; skipping equivalence gate.")
            return
        rng = np.random.RandomState(cfg.RANDOM_SEED)
        sample = rng.choice(
            legacy, size=min(QUAT_LABEL_GATE_INSTANCES, len(legacy)),
            replace=False)
        worst = 0.0
        bad = 0
        for i in sample:
            old = np.load(
                self.dataset_dicts[i]['inst_infos']['quat_file']).astype(
                    np.float64)
            new = labels[i].astype(np.float64)
            diff = float(np.abs(new - old).max())
            worst = max(worst, diff)
            if not np.allclose(new, old, rtol=1e-5, atol=1e-6):
                bad += 1
        logger.info(
            "quat label equivalence: max_abs_diff=%.3e over %d instances",
            worst, len(sample))
        if bad > 0:
            raise AssertionError(
                "quat label equivalence gate failed for {}/{} instances "
                "(max_abs_diff={:.3e})".format(
                    bad, len(sample), worst))

    def get_info(self):
        return {
            'depth_min': self.depth_min,
            'depth_max': self.depth_max,
            'num_objects': self.num_objects}

    def __len__(self):
        return len(self.dataset_dicts)

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        data_dict = self.dataset_dicts[idx]
        batch = self.read_data(data_dict, idx)
        return batch

    def read_data(self, dataset_dict, idx):
        inst_infos = dataset_dict['inst_infos']
        obj_id = inst_infos['obj_id']
        scene_id = inst_infos['scene_id']
        image_id = inst_infos['im_id']

        image_file = inst_infos["image_file"]

        image = file_io.imread(image_file, 'color', self.img_format)
        image = image.astype(np.float32)
        im_H, im_W = image.shape[:2]
        if cfg.CACHE_MASK:
            mask = misc.cocosegm2mask(inst_infos["mask_file"], im_H, im_W)
        else:
            mask = file_io.imread(inst_infos["mask_file"], "unchanged").astype(bool).astype(np.uint8)
        ### RGB augmentation ###
        if (self.split == 'train' or self.split == 'finetune') and np.random.rand() < self.COLOR_AUG_PROB:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()
            augmented = self.color_augmentor({'image': image_t, 'mask': mask_t})
            image = (augmented['image'] * 255.0).permute(1, 2, 0).numpy().astype(np.float32)
            mask = augmented['mask'][0].numpy().astype(np.uint8)

        obj_R = inst_infos['rotation'].astype("float32").reshape(3, 3)
        obj_t = inst_infos['translation'].astype("float32").reshape(3,)
        cam_K = inst_infos['intrinsics'].astype("float32")

        bx, by, bw, bh = inst_infos["bbox"]
        bbox_xyxy = np.array([bx, by, bx+bw, by+bh])

        if self.split == 'train' or self.split == 'finetune':
            bbox_center, bbox_scale, bbox_loc = misc.aug_bbox_DZI(
                bbox_xyxy, im_H, im_W,
                scale_ratio=self.DZI_SCALE_RATIO,
                shift_ratio=self.DZI_SHIFT_RATIO,
                pad_scale=self.DZI_PAD_SCALE,
            )  # Dynamic zoom-in see the paper GDR-Net
        else:
            x1, y1, x2, y2 = bbox_xyxy.copy()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bbox_center = np.array([cx, cy])
            bbox_scale = self.DZI_PAD_SCALE * max(y2 - y1, x2 - x1)
            hr = (y2 - y1) / bbox_scale
            wr = (x2 - x1) / bbox_scale
            bbox_loc = np.array([0.5 - wr/2, 0.5 - hr/2, 0.5 + wr/2, 0.5 + hr/2])

        obj_inst_count = dataset_dict['obj_inst_count']
        rot_index = 0
        if self.split == 'train': #### randomly replace the background if an image contains multiple instances of the same object ####
            if obj_inst_count[obj_id] > 2 and np.random.rand() < self.CHANGE_BG_PROB:
                image = self.replace_bg(image.copy(), mask)  # multiple instances in a ROI
            rot_index = np.random.randint(4)
        elif self.split == 'finetune':
            if np.random.rand() < self.CHANGE_BG_PROB:
                image = self.replace_bg(image.copy(), mask)  # multiple instances in a ROI
            rot_index = np.random.randint(4)

        rot_rad = rot_index * np.pi / 2
        Rz = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                       [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        obj_R = Rz @ obj_R
        obj_t = Rz @ obj_t
        roi_mask = misc.crop_resize_by_warp_affine(
            mask, bbox_center, bbox_scale, self.mask_size, cam_K, rot_rad, interpolation='bilinear'
        ).squeeze(0)  # HxW
        roi_mask = (roi_mask > 0.5).float()

        roi_img = misc.crop_resize_by_warp_affine(
            image, bbox_center, bbox_scale, self.rgb_size, cam_K, rot_rad, interpolation='bilinear'
        ) / 255.0  # HxWx3 -> 3xHxW

        T_rot = cam_K @ Rz @ np.linalg.inv(cam_K)
        center_hom = T_rot @ np.array(
            [*bbox_center, 1.0], dtype=np.float32)
        bbox_center = center_hom[:2]

        fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
        fov = np.array([(bbox_center[0] - cx) / fx, (bbox_center[1] - cy) / fy, bbox_scale / fx])

        T_img2roi = misc.transform_to_local_ROIcrop(bbox_center=bbox_center, bbox_scale=bbox_scale, zoom_scale=self.rgb_size)
        roi_camK = T_img2roi.numpy() @ cam_K

        Tz = np.array([[1.0, 0.0, -0.5],
                       [0.0, 1.0, -0.5],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        T_loc = np.linalg.inv(Tz) @ Rz @ Tz
        bbox_loc = utils.transform_bounding_box(bbox_loc, T_loc)
        bbox_map = utils.make_roi(torch.as_tensor(bbox_loc, dtype=torch.float32), self.rgb_size).unsqueeze(0)

        quat_bin = np.load(self.quat_label_path, mmap_mode='r')

        return {
            'roi_image': torch.as_tensor(roi_img, dtype=torch.float32).contiguous(),
            'bbox_map': torch.as_tensor(bbox_map, dtype=torch.float32),
            'roi_camK': torch.as_tensor(roi_camK, dtype=torch.float32).squeeze(),
            'fov': torch.as_tensor(fov, dtype=torch.float32),
            'obj_cls': torch.as_tensor(self.dataset_id2cls[obj_id], dtype=torch.int64),
            'roi_obj_R': torch.as_tensor(obj_R, dtype=torch.float32),
            'roi_obj_t': torch.as_tensor(obj_t, dtype=torch.float32),
            'roi_mask': torch.as_tensor(roi_mask, dtype=torch.float32).contiguous(),
            'quat_bin': torch.as_tensor(
                np.array(quat_bin[idx][rot_index], dtype=np.float32)),
            'obj_id': int(obj_id),
            'scene_id': int(scene_id),
            'image_id': int(image_id),
        }

    @misc.lazy_property
    def _bg_img_paths(self):
        bg_type = self.BG_TYPE
        bg_root = self.BG_ROOT
        bg_num = self.NUM_BG_IMGS

        logger.info("get bg image paths")
        hashed_file_name = hashlib.md5(
            ("{}_{}_{}_get_bg_imgs".format(bg_root, bg_num, bg_type)).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(".cache/bg_paths_{}_{}.pkl".format(bg_type, hashed_file_name))
        Path(osp.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
        if osp.exists(cache_path):
            logger.info("get bg_paths from cache file: {}".format(cache_path))
            bg_img_paths = file_io.load(cache_path)
            logger.info("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            return bg_img_paths

        logger.info("building bg imgs cache {}...".format(bg_type))
        assert osp.exists(bg_root), f"BG ROOT: {bg_root} does not exist"
        if bg_type == "coco":
            img_paths = [
                osp.join(bg_root, fn.name) for fn in os.scandir(bg_root) if ".png" in fn.name or "jpg" in fn.name
            ]
        elif bg_type == "VOC_table":  # used in original deepim
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            VOC_image_set_dir = osp.join(VOC_root, "ImageSets/Main")
            VOC_bg_list_path = osp.join(VOC_image_set_dir, "diningtable_trainval.txt")
            with open(VOC_bg_list_path, "r") as f:
                VOC_bg_list = [
                    line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                ]
            img_paths = [osp.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        elif bg_type == "VOC":
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            img_paths = [
                osp.join(VOC_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        elif bg_type == "SUN2012":
            img_paths = [
                osp.join(bg_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        else:
            raise ValueError(f"BG_TYPE: {bg_type} is not supported")
        assert len(img_paths) > 0, len(img_paths)

        num_bg_imgs = min(len(img_paths), bg_num)
        bg_img_paths = np.random.choice(img_paths, num_bg_imgs)

        file_io.dump(bg_img_paths, cache_path)
        logger.info("num bg imgs: {}".format(len(bg_img_paths)))
        assert len(bg_img_paths) > 0
        return bg_img_paths

    def trunc_mask(self, mask):
        # return the bool truncated mask
        mask = mask.copy().astype(bool)
        nonzeros = np.nonzero(mask.astype(np.uint8))
        x1, y1 = np.min(nonzeros, axis=1)
        x2, y2 = np.max(nonzeros, axis=1)
        c_h = 0.5 * (x1 + x2)
        c_w = 0.5 * (y1 + y2)
        rnd = random.random()
        if rnd < 0.2:  # block upper
            c_h_ = int(random.uniform(x1, c_h))
            mask[:c_h_, :] = False
        elif rnd < 0.4:  # block bottom
            c_h_ = int(random.uniform(c_h, x2))
            mask[c_h_:, :] = False
        elif rnd < 0.6:  # block left
            c_w_ = int(random.uniform(y1, c_w))
            mask[:, :c_w_] = False
        elif rnd < 0.8:  # block right
            c_w_ = int(random.uniform(c_w, y2))
            mask[:, c_w_:] = False
        else:
            pass
        return mask

    def replace_bg(self, im, im_mask, return_mask=False, truncate_fg=False,
                   synthesize_blending_artifacts=True):
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self._bg_img_paths) - 1)
        filename = self._bg_img_paths[ind]

        if self.BG_KEEP_ASPECT_RATIO:
            bg_img = self.get_bg_image(filename, H, W)
        else:
            bg_img = self.get_bg_image_v2(filename, H, W)

        if synthesize_blending_artifacts:
            idx = random.randint(0, len(self.dataset_dicts) - 1)
            sample_dict = self.dataset_dicts[idx]
            sample_info = sample_dict['inst_infos']
            sample_mask = misc.cocosegm2mask(
                sample_info['mask_file'], *im_mask.shape[:2])
            bx, by, bw, bh = sample_info['bbox']
            bbox_center = np.array([bx + bw / 2, by + bh / 2])
            y, x = np.where(im_mask)
            if y.size > 0:
                xmin, xmax = np.min(x), np.max(x)
                ymin, ymax = np.min(y), np.max(y)
                mask_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
                shift = mask_center - bbox_center
                ys, xs = np.where(sample_mask)
                yt = np.minimum(np.maximum(ys + round(shift[1]), 0), H-1)
                xt = np.minimum(np.maximum(xs + round(shift[0]), 0), W-1)
                bg_img[yt, xt] = bg_img[ys, xs]

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            logger.warning("bad background image: {}".format(filename))

        mask = im_mask.copy().astype(bool)
        if truncate_fg:
            mask = self.trunc_mask(im_mask)
        mask_bg = ~mask
        bg_img = bg_img.astype(np.float32)
        im[mask_bg] = bg_img[mask_bg]
        if return_mask:
            return im, mask  # bool fg mask
        else:
            return im

    def get_bg_image(self, filename, imH, imW, channel=3):
        """keep aspect ratio of bg during resize target image size:

        imHximWxchannel.
        """
        target_size = min(imH, imW)
        max_size = max(imH, imW)
        real_hw_ratio = float(imH) / float(imW)
        bg_image = file_io.imread(filename, 'color', self.img_format)
        bg_h, bg_w, bg_c = bg_image.shape
        bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
        if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                else:
                    bg_image_crop = bg_image
            else:
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                else:
                    bg_image_crop = bg_image
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
        bg_image_resize_0 = misc.resize_short_edge(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
        return bg_image_resize

    def get_bg_image_v2(self, filename, imH, imW, channel=3):
        _bg_img = file_io.imread(filename, 'color', self.img_format)
        try:
            # randomly crop a region as background
            bw = _bg_img.shape[1]
            bh = _bg_img.shape[0]
            x1 = np.random.randint(0, int(bw / 3))
            y1 = np.random.randint(0, int(bh / 3))
            x2 = np.random.randint(int(2 * bw / 3), bw)
            y2 = np.random.randint(int(2 * bh / 3), bh)
            bg_img = cv2.resize(_bg_img[y1:y2, x1:x2], (imW, imH),
                                interpolation=cv2.INTER_LINEAR)
        except:
            bg_img = np.zeros((imH, imW, 3), dtype=np.uint8)
            logger.warning("bad background image: {}".format(filename))
        return bg_img
