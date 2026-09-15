"""Regression guard for the MRC-Net refactor (Phase 0).

`capture` runs the current model on a fixed synthetic batch and stores every
prediction tensor (plus the state_dict key list) under
``.cache/regression/reference.pt``.  `check` re-runs the identical batch and
verifies that the state_dict keys and all prediction tensors are unchanged.

The renderer loads CAD meshes from ``/mnt/d/6DPose/usprobe/models`` so this
script only works in an environment where that path exists (WSL + conda env
``mrcnet``).
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytorch3d.io  # noqa: F401  (ensure pytorch3d.io submodule attribute)
import config as cfg
import models
import utils

CHECKPOINT = os.path.join(ROOT, 'chk_usprobe', 'mrcnet_ycb_usprobe.pth')
REF_DIR = os.path.join(ROOT, '.cache', 'regression')
REF_PATH = os.path.join(REF_DIR, 'reference.pt')

DATASET = 'usprobe'
N_DECODERS = 4
DEPTH_MIN = 0.010
DEPTH_MAX = 2.000
BATCH = 8
IMG_SIZE = cfg.INPUT_IMG_SIZE

# Every prediction tensor observed on the unmodified forward pass.
PRED_KEYS = [
    'roi_mask', 'roi_mask_synt', 'roi_obj_R', 'quat_bin', 'quat_res',
    'depth_bin', 'depth_res', 'trans_xy', 'trans_logits', 'trans_res',
    'translation', 'render', 'mask_synt',
]


def setup_determinism():
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_inputs(device):
    """Deterministic synthetic batch (independent of model-init RNG use)."""
    torch.manual_seed(cfg.RANDOM_SEED)
    roi_image = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE).to(device)

    bbox_loc = torch.tensor([
        [0.20, 0.25, 0.80, 0.75],
        [0.15, 0.20, 0.70, 0.85],
        [0.30, 0.10, 0.90, 0.60],
        [0.10, 0.30, 0.60, 0.90],
        [0.25, 0.25, 0.75, 0.75],
        [0.18, 0.22, 0.82, 0.78],
        [0.22, 0.15, 0.78, 0.65],
        [0.12, 0.28, 0.68, 0.88],
    ], dtype=torch.float32)
    bbox_map = torch.stack(
        [utils.make_roi(bbox_loc[i], IMG_SIZE) for i in range(BATCH)],
        dim=0).unsqueeze(1).to(device)

    obj_cls = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64, device=device)

    fov = torch.tensor([
        [0.00, 0.00, 0.50],
        [0.05, -0.03, 0.45],
        [-0.04, 0.06, 0.55],
        [0.02, 0.01, 0.40],
        [0.00, 0.00, 0.50],
        [0.05, -0.03, 0.45],
        [-0.04, 0.06, 0.55],
        [0.02, 0.01, 0.40],
    ], dtype=torch.float32).to(device)

    K = torch.tensor([[512., 0., 128.],
                      [0., 512., 128.],
                      [0., 0., 1.]], dtype=torch.float32)
    roi_camK = K.unsqueeze(0).repeat(BATCH, 1, 1).to(device)

    inputs = torch.cat([roi_image, bbox_map], dim=1)
    aux = {'obj_cls': obj_cls, 'fov': fov, 'intrinsics': roi_camK}
    return inputs, aux


def build_model(device):
    model_config = cfg.ModelConfig.from_dataset_config(DATASET)
    model = models.MRCNet(model_config).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['network'], strict=True)
    model.eval()
    return model


def run_forward(model, device):
    inputs, aux = build_inputs(device)
    with torch.no_grad():
        try:
            torch.use_deterministic_algorithms(True)
            preds = model(inputs, aux)
            mode = 'strict'
        except RuntimeError:
            torch.use_deterministic_algorithms(True, warn_only=True)
            preds = model(inputs, aux)
            mode = 'warn'
    return preds, mode


def capture(device):
    setup_determinism()
    model = build_model(device)
    preds, mode = run_forward(model, device)
    os.makedirs(REF_DIR, exist_ok=True)

    state_keys = sorted(model.state_dict().keys())
    tensors = {}
    shapes = {}
    for key in PRED_KEYS:
        assert key in preds, 'missing prediction key: {}'.format(key)
        value = preds[key]
        assert torch.is_tensor(value), '{} is not a tensor'.format(key)
        tensors[key] = value.detach().cpu().clone()
        shapes[key] = tuple(value.shape)

    torch.save({
        'state_keys': state_keys,
        'tensors': tensors,
        'shapes': shapes,
        'det_mode': mode,
    }, REF_PATH)
    print('Captured reference to {} ({} keys, determinism={})'.format(
        REF_PATH, len(state_keys), mode))


def check(device):
    setup_determinism()
    if not os.path.exists(REF_PATH):
        print('No reference at {}; run `capture` first.'.format(REF_PATH))
        sys.exit(2)
    ref = torch.load(REF_PATH, map_location='cpu', weights_only=True)

    model = build_model(device)
    preds, mode = run_forward(model, device)

    failures = []
    state_keys = sorted(model.state_dict().keys())
    if state_keys != ref['state_keys']:
        only_ref = set(ref['state_keys']) - set(state_keys)
        only_new = set(state_keys) - set(ref['state_keys'])
        failures.append(
            'state_dict keys differ; removed={} added={}'.format(
                sorted(only_ref), sorted(only_new)))
    else:
        print('state_dict keys: OK ({} keys)'.format(len(state_keys)))

    if mode != ref['det_mode']:
        print('WARNING: determinism mode changed {} -> {}'.format(
            ref['det_mode'], mode))

    for key in PRED_KEYS:
        ref_t = ref['tensors'][key]
        new_t = preds[key].detach().cpu()
        if tuple(new_t.shape) != ref['shapes'][key]:
            failures.append('{} shape {} != reference {}'.format(
                key, tuple(new_t.shape), ref['shapes'][key]))
            continue
        diff = (new_t.float() - ref_t.float()).abs()
        max_diff = float(diff.max()) if diff.numel() else 0.0
        ok = torch.allclose(new_t, ref_t, rtol=1e-4, atol=1e-5)
        status = 'OK' if ok else 'FAIL'
        print('{:<14} {} max_abs_diff={:.3e}'.format(key, status, max_diff))
        if not ok:
            failures.append('{} max_abs_diff={:.3e}'.format(key, max_diff))

    if failures:
        print('\nREGRESSION GUARD FAILED:')
        for f in failures:
            print('  - {}'.format(f))
        sys.exit(1)
    print('\nRegression guard passed.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['capture', 'check'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available; falling back to CPU (renderer/guard may be slow).')
        device = 'cpu'

    if args.mode == 'capture':
        capture(device)
    else:
        check(device)


if __name__ == '__main__':
    main()