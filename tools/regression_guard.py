"""Regression guard for the MRC-Net refactor (Phase 0).

`capture` runs the current model on a fixed synthetic batch and stores every
prediction tensor (plus the state_dict key list) under
``.cache/regression/reference.pt``.  `check` re-runs the identical batch and
verifies that the state_dict keys and all prediction tensors are unchanged.

The renderer loads CAD meshes from ``config.DATASET_ROOT/usprobe/models`` so
this script only works where that path exists (native Windows ``.venv`` or the
legacy WSL/conda env).

A deterministic ``targets`` dict is also run through ``forward`` and every
loss value is captured alongside the prediction tensors, giving a second
bit-for-bit gate for the loss refactor (the prediction check alone runs
without targets).
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

# Every loss value produced by ``forward`` when ``targets`` is provided.
LOSS_KEYS = [
    'quat_clf', 'quat_reg', 'depth_clf', 'txty_clf', 'txty_reg', 'tz_reg',
    'mask_visib',
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


def build_targets(device, aux):
    """Deterministic targets dict for the loss reference.

    Only the keys consumed by ``MRCNet._compute_loss`` (plus the renderer
    conditioning shared with ``aux``) are provided; ``quat_init``, ``trans_2d``
    and ``depth_id`` are filled in by ``forward`` itself from its own
    predictions.
    """
    torch.manual_seed(cfg.RANDOM_SEED + 1)
    quat = torch.randn(BATCH, 4)
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    roi_obj_R = utils.quaternion_to_rotation(quat)
    roi_obj_t = torch.stack([
        0.1 * torch.randn(BATCH),
        0.1 * torch.randn(BATCH),
        0.2 + 0.2 * torch.rand(BATCH)], dim=-1)
    roi_mask = (torch.rand(
        BATCH, cfg.MASK_SIZE, cfg.MASK_SIZE) > 0.5).float()
    quat_bin = torch.softmax(torch.randn(BATCH, cfg.N_POSE_BIN), dim=-1)
    targets = {
        'roi_obj_R': roi_obj_R,
        'roi_obj_t': roi_obj_t,
        'roi_mask': roi_mask,
        'quat_bin': quat_bin,
        'roi_camK': aux['intrinsics'].clone(),
        'obj_cls': aux['obj_cls'].clone(),
        'fov': aux['fov'].clone(),
    }
    return {key: value.to(device) for key, value in targets.items()}


def build_model(device):
    model_config = cfg.ModelConfig.from_dataset_config(DATASET)
    model = models.MRCNet(model_config).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['network'], strict=True)
    model.eval()
    return model


def run_forward(model, device, with_targets=False):
    inputs, aux = build_inputs(device)
    with torch.no_grad():
        try:
            torch.use_deterministic_algorithms(True)
            if with_targets:
                preds = model(inputs, aux, build_targets(device, aux))
            else:
                preds = model(inputs, aux)
            mode = 'strict'
        except RuntimeError:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if with_targets:
                preds = model(inputs, aux, build_targets(device, aux))
            else:
                preds = model(inputs, aux)
            mode = 'warn'
    return preds, mode


def capture(device):
    setup_determinism()
    model = build_model(device)
    preds, mode = run_forward(model, device)
    loss_preds, loss_mode = run_forward(model, device, with_targets=True)
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

    assert 'losses' in loss_preds, 'forward(with targets) returned no losses'
    losses = {}
    for key in LOSS_KEYS:
        assert key in loss_preds['losses'], \
            'missing loss key: {}'.format(key)
        losses[key] = loss_preds['losses'][key].detach().cpu().clone()

    torch.save({
        'state_keys': state_keys,
        'tensors': tensors,
        'shapes': shapes,
        'det_mode': mode,
        'losses': losses,
        'loss_det_mode': loss_mode,
    }, REF_PATH)
    print('Captured reference to {} ({} state keys, determinism={} and {} '
          'with targets)'.format(REF_PATH, len(state_keys), mode, loss_mode))
    for key in LOSS_KEYS:
        print('  loss {:<11} {:.6e}'.format(key, float(losses[key])))


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

    if 'losses' in ref:
        loss_preds, loss_mode = run_forward(model, device, with_targets=True)
        if loss_mode != ref.get('loss_det_mode', loss_mode):
            print('WARNING: loss determinism mode changed {} -> {}'.format(
                ref.get('loss_det_mode'), loss_mode))
        if 'losses' not in loss_preds:
            failures.append('forward(with targets) returned no losses')
        else:
            for key in LOSS_KEYS:
                if key not in ref['losses']:
                    failures.append('loss {} missing from reference'.format(key))
                    continue
                if key not in loss_preds['losses']:
                    failures.append('loss {} missing from forward'.format(key))
                    continue
                ref_l = ref['losses'][key].float()
                new_l = loss_preds['losses'][key].detach().cpu().float()
                max_diff = float((new_l - ref_l).abs().max())
                ok = torch.allclose(new_l, ref_l, rtol=1e-5, atol=1e-6)
                status = 'OK' if ok else 'FAIL'
                print('loss/{:<9} {} max_abs_diff={:.3e}'.format(
                    key, status, max_diff))
                if not ok:
                    failures.append('loss {} max_abs_diff={:.3e}'.format(
                        key, max_diff))

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