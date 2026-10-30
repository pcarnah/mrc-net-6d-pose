import os
from dataclasses import dataclass

INPUT_SIZE = 256
CELL_SIZE = 64
MASK_SIZE = 64
USE_6D = True
N_POSE_BIN = 4608  # number of bins for 3D orientation

WEIGHT_TRANS_CLF = 2.0
WEIGHT_DEPTH_CLF = 2.0  # 1.0 for hard labels
WEIGHT_QUAT_CLF = 0.5
WEIGHT_MASK = 10.0
WEIGHT_QUAT_REG = 1.0
WEIGHT_TRANS_REG = 1.0
WEIGHT_DEPTH_REG = 0.2

# Deep supervision of the iterative refinement loop (`MRCNet.forward`).
# None -> every refinement iteration (including iteration 0) is weighted 1.0;
# otherwise a list/tuple of per-iteration weights of length n_refine_iters.
REFINE_ITER_WEIGHTS = None
# Opt-in gradient checkpointing for the synthetic backbone+decoder branch.
REFINE_GRAD_CKPT = False

# Two-view fusion variants for the stereo residual-regression branch. All of
# them feed the *unchanged* shared regressor with a left-frame feature, so the
# ablation isolates the fusion operation rather than the camera geometry:
#   cross_view : learned ConvGnReLU(cat(left, right))  (default, main model)
#   mean       : parameter-free 0.5 * (left + right)   (alternative/simple)
#   left_only  : left reg_pool only                    (second-view ablation)
#   right_only : right reg_pool only                   (second-view ablation)
# `left_only`/`right_only` still render both views (so the comparison holds the
# input pipeline fixed) but discard the unused view's features.
FUSION_MODES = ('cross_view', 'mean', 'left_only', 'right_only')
DEFAULT_FUSION_MODE = 'cross_view'

RANDOM_SEED = 2022      # random seed
INPUT_IMG_SIZE = 256    # the input image size of network
OUTPUT_MASK_SIZE = 64   # the output mask size of network
Tz_BINS_NUM = 1000      # the number of discretized depth bins
POSE_SIGMA = 0.03  # standard deviation of quaternion bin distribution
DEPTH_SIGMA = 0.5     # standard deviation of depth Gaussian distribution
TRANS_SIGMA = 10 / INPUT_SIZE  # standard deviation of noise in 2D center

# CAD meshes with more faces than this are quadric-decimated at registry build
# time (pyvista/VTK).  The pytorch3d rasterizer cost scales with the face count
# (naive path) and needs a large ``max_faces_per_bin`` (binned path), so the
# dense stereobj-1m models (up to 70k faces) are reduced here.  Keep this at or
# below 10000: the rasterizer's default ``max_faces_per_bin`` is
# ``max(10000, faces/5)``, so a decimated mesh can then never overflow a bin.
MAX_MESH_FACES = 10000

def _env_path(name, default):
    """Root path from ``MRC_<NAME>`` / ``<NAME>`` env vars, else ``default``.

    Lets a cluster job point the data/eval roots at a staging dir (e.g.
    ``$SLURM_TMPDIR``) without editing this file.
    """
    return os.environ.get('MRC_' + name, os.environ.get(name, default))


DATASET_ROOT = _env_path('DATASET_ROOT', '../datasets')
# EVAL_ROOT follows DATASET_ROOT unless overridden explicitly.
EVAL_ROOT = os.environ.get(
    'MRC_EVAL_ROOT', os.environ.get('EVAL_ROOT', DATASET_ROOT))
# DATASET_ROOT = "/mnt/d/6DPose"
# EVAL_ROOT = "/mnt/d/6DPose"
VOC_BG_ROOT = _env_path('VOC_BG_ROOT', './data/VOCdevkit/VOC2012')


def set_roots(dataset_root=None, eval_root=None):
    """Override the dataset/eval roots for this process (CLI ``--dataset_root``
    / ``--eval_root``).  Values are also exported to ``MRC_DATASET_ROOT`` /
    ``MRC_EVAL_ROOT`` so spawned DataLoader workers and child processes see the
    same roots.  Returns ``(DATASET_ROOT, EVAL_ROOT)``.
    """
    global DATASET_ROOT, EVAL_ROOT
    if dataset_root is not None:
        DATASET_ROOT = dataset_root
        os.environ['MRC_DATASET_ROOT'] = dataset_root
    if eval_root is not None:
        EVAL_ROOT = eval_root
        os.environ['MRC_EVAL_ROOT'] = eval_root
    return DATASET_ROOT, EVAL_ROOT

END_LR = 5e-6
START_LR = 2e-4
END_LR_FT = 5e-7
START_LR_FT = 5e-6
DECAY_WEIGHT = 1e-3

USE_CACHE = True
CACHE_MASK = True
ZOOM_PAD_SCALE = 1.5
ZOOM_SCALE_RATIO = 0.25
ZOOM_SHIFT_RATIO = 0.25   # center shift
COLOR_AUG_PROB = 0.8
CHANGE_BG_PROB = 0.8      # the prob for changing the background

RZ_ROTATION_AUG = False

DATASET_CONFIG = {
    'tless': {
        'width': 720,
        'height': 540,
        'color_type': 'rgb',
        'Tz_near': 0.050,
        'Tz_far': 1.700,
        'num_class': 30,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(30)},
        'id2cls': {v+1: v for v in range(30)},  # from object_id to class_index
        'model_folders': {
            'train_pbr': 'models_cad',
            'train_primesense': 'models_cad',
            'test_primesense': 'models_cad'},
        'train_set': ['train_pbr'],
        'finetune_set': ['train_primesense'],
        'test_set': ['test_primesense'],
    },  # √
    'itodd': {
        'width': 1280,
        'height': 960,
        'color_type': 'gray',
        'Tz_near': 0.01,
        'Tz_far': 1.85,
        'num_class': 28,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(28)},
        'id2cls': {v+1: v for v in range(28)},
        'model_folders': {
            'train_pbr': 'models',
            'val': 'models',
            'test': 'models'},
        'train_set': ['train_pbr'],
        'test_set': ['val'],
    },
    'ycbv': {
        'width': 640,
        'height': 480,
        'color_type': 'rgb',
        'Tz_near': 0.030,
        'Tz_far': 2.000,
        'num_class': 21,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(21)},
        'id2cls': {v+1: v for v in range(21)},  # from object_id to class_index
        'model_folders': {
            'train_pbr': 'models',
            'train_real': 'models',
            'train_synt': 'models',
            'test': 'models'},
        'train_set': ['train_pbr'],
        'finetune_set': ['train_real', 'train_synt'],
        'test_set': ['test'],
    },  # √
    'lmo': {
        'width': 640,
        'height': 480,
        'color_type': 'rgb',
        'Tz_near': 0.010,
        'Tz_far': 2.150,
        'num_class': 8,
        'id2mod': {v: "obj_{:02d}".format(v) for v in [
            1, 5, 6, 8, 9, 10, 11, 12]},
        'id2cls': {v: i for i, v in enumerate([
            1, 5, 6, 8, 9, 10, 11, 12])},  # from object_id to class_index
        'model_folders': {
            'train_pbr': 'models',
            'test': 'models'},
        'train_set': ['train_pbr'],
        'finetune_set': ['train'],
        'test_set': ['test'],
    },  # √
    'ycbv-ds': {
        'width': 640,
        'height': 480,
        'color_type': 'rgb',
        'bop_dataset': 'ycbv',
        'Tz_near': 0.030,
        'Tz_far': 2.000,
        'num_class': 21,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(21)},
        'id2cls': {v+1: v for v in range(21)},  # from object_id to class_index
        'model_folders': {
            'train_pbr': 'models',
            'train_pbr_left': 'models',
            'test': 'models_eval',
            'test~left_pbr': 'models_eval'},
        'train_set': ['train_pbr_left'],
        # NOTE: train_pbr_left_extras exists but is empty on disk; keep
        # finetune on train_pbr_left until it is populated.
        'finetune_set': ['train_pbr_left'],
        'test_set': ['test~left_pbr'],
    },
    'stereobj-1m': {
        # WebDataset of stereo pairs (webds/{train,val,test}-*.tar).
        # Each sample is a 2000x1000 side-by-side stereo pair: left eye at
        # x in [0, 1000), right eye at x in [1000, 2000). Mono mode treats
        # each eye as an independent scene.
        'width': 1000,
        'height': 1000,
        'color_type': 'rgb',
        'Tz_near': 0.100,
        'Tz_far': 2.000,
        'num_class': 18,
        'id2mod': {name: name for name in [
            'centrifuge_tube', 'microplate', 'pipette_0.5_10',
            'pipette_10_100', 'pipette_100_1000', 'sterile_tip_rack_10',
            'sterile_tip_rack_200', 'sterile_tip_rack_1000',
            'tube_rack_1.5_2_ml', 'tube_rack_50_ml', 'blade_razor',
            'hammer', 'needle_nose_pliers', 'screwdriver', 'side_cutters',
            'tape_measure', 'wire_stripper', 'wrench']},
        'id2cls': {name: i for i, name in enumerate([
            'centrifuge_tube', 'microplate', 'pipette_0.5_10',
            'pipette_10_100', 'pipette_100_1000', 'sterile_tip_rack_10',
            'sterile_tip_rack_200', 'sterile_tip_rack_1000',
            'tube_rack_1.5_2_ml', 'tube_rack_50_ml', 'blade_razor',
            'hammer', 'needle_nose_pliers', 'screwdriver', 'side_cutters',
            'tape_measure', 'wire_stripper', 'wrench'])},
        'model_format': 'stereobj',
        'model_folders': {
            'train': 'objects',
            'val': 'objects',
            'test': 'objects'},
        'loader': 'webds',
        'webds_dir': 'webds',
        'train_set': ['train'],
        'finetune_set': ['train'],
        # webds test shards carry images only (no labels); the annotated
        # val shards are the eval split. Both splits resolve to the val
        # shards; split='val' exposes them directly too.
        'val_set': ['val'],
        'test_set': ['val'],
        # 'mono': each eye is an independent scene (PoseSample stays
        # single-view). 'stereo': one sample per instance annotated on both
        # eyes; __getitem__ returns the left-eye PoseSample plus the right-eye
        # inputs (roi_image_right, bbox_map_right, roi_camK_right, fov_right)
        # and the per-sample rig transform T_right_ref. Targets stay in the
        # left-camera frame. Only the residual-regression branch fuses both
        # views (stereo_fusion=True); the classification bootstrap stays mono.
        'stereo_mode': 'stereo',
        'stereo_fusion': True,
        # Fusion operation; overridable per run via --fusion (see FUSION_MODES).
        'fusion_mode': 'cross_view',
    },
    'usprobe': {
        'width': 1280,
        'height': 720,
        'color_type': 'gray',
        'Tz_near': 0.010,
        'Tz_far': 2.000,
        'num_class': 4,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(4)},
        'id2cls': {v+1: v for v in range(4)},  # from object_id to class_index
        'model_folders': {
            'train': 'models',
            'train_pbr': 'models',
            'test': 'models_eval'
        },
        'train_set': ['train'],
        'test_set': ['test'],
    }
}


def validate_dataset_config():
    """Validate every DATASET_CONFIG entry; raise ValueError on the first
    inconsistency found. Cheap and dict-only, safe to call at import time.
    """
    required_keys = ('width', 'height', 'color_type', 'Tz_near', 'Tz_far',
                     'num_class', 'id2mod', 'id2cls', 'model_folders',
                     'train_set', 'test_set')
    for name, ds_cfg in DATASET_CONFIG.items():
        missing = [k for k in required_keys if k not in ds_cfg]
        if missing:
            raise ValueError(
                'DATASET_CONFIG[{}]: missing required keys {}'.format(
                    name, missing))
        if ds_cfg['color_type'] not in ('rgb', 'gray'):
            raise ValueError(
                "DATASET_CONFIG[{}]: color_type must be 'rgb' or 'gray', "
                "got {!r}".format(name, ds_cfg['color_type']))
        if len(ds_cfg['id2cls']) != ds_cfg['num_class']:
            raise ValueError(
                'DATASET_CONFIG[{}]: len(id2cls)={} != num_class={}'.format(
                    name, len(ds_cfg['id2cls']), ds_cfg['num_class']))
        if set(ds_cfg['id2cls']) != set(ds_cfg['id2mod']):
            raise ValueError(
                'DATASET_CONFIG[{}]: id2cls keys {} != id2mod keys {}'.format(
                    name, sorted(ds_cfg['id2cls']), sorted(ds_cfg['id2mod'])))


validate_dataset_config()


@dataclass(frozen=True)
class ModelConfig:
    """Model-shape hyperparameters, saved inside checkpoints so inference can
    rebuild the network without deriving globals from the dataset name."""
    dataset: str
    n_decoders: int
    depth_min: float
    depth_max: float
    n_depth_bin: int
    n_pose_bin: int = N_POSE_BIN
    use_6d: bool = USE_6D
    cell_size: int = CELL_SIZE
    input_img_size: int = INPUT_IMG_SIZE
    # Two-view residual fusion (stereobj-1m stereo mode). Defaults to False so
    # old checkpoint model_config dicts still construct and mono models keep a
    # bit-for-bit identical state_dict.
    stereo_fusion: bool = False
    # Fusion operation used when ``stereo_fusion`` is True; see FUSION_MODES.
    # Defaults to the main model's learned cross-view concat so old stereo
    # checkpoints (which predate this field) rebuild identically.
    fusion_mode: str = DEFAULT_FUSION_MODE

    @classmethod
    def from_dataset_config(cls, dataset, stereo_mode=None, fusion=None):
        dataset_cfg = DATASET_CONFIG[dataset]
        stereo = dataset_cfg.get('stereo_fusion', False)
        if stereo_mode is not None:
            stereo = (stereo_mode == 'stereo')
        fusion_mode = dataset_cfg.get('fusion_mode', DEFAULT_FUSION_MODE)
        if fusion is not None:
            fusion_mode = fusion
        return cls(
            dataset=dataset,
            n_decoders=dataset_cfg['num_class'],
            depth_min=dataset_cfg['Tz_near'],
            depth_max=dataset_cfg['Tz_far'],
            n_depth_bin=Tz_BINS_NUM,
            stereo_fusion=stereo,
            fusion_mode=fusion_mode)


def set_runtime_overrides(dataset, stereo_mode=None, fusion=None):
    """Apply per-run overrides to ``DATASET_CONFIG[dataset]`` in place.

    SLURM jobs are one configuration per process, so mutating the module
    global is the least invasive way to make mono/stereo and the fusion
    ablation selectable from the command line without editing this file.
    Returns the mutated dict.
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError('unknown dataset {!r}'.format(dataset))
    ds_cfg = DATASET_CONFIG[dataset]
    if stereo_mode is not None:
        if stereo_mode not in ('mono', 'stereo'):
            raise ValueError(
                "stereo_mode must be 'mono' or 'stereo', got {!r}".format(
                    stereo_mode))
        ds_cfg['stereo_mode'] = stereo_mode
        ds_cfg['stereo_fusion'] = (stereo_mode == 'stereo')
    if fusion is not None:
        if fusion not in FUSION_MODES:
            raise ValueError('fusion must be one of {}, got {!r}'.format(
                FUSION_MODES, fusion))
        ds_cfg['fusion_mode'] = fusion
    return ds_cfg
