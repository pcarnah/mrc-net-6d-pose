"""Smoke test for the stereo refinement path (``stereo_fusion=True``).

Mono behaviour is gated separately by ``tools/regression_guard.py`` and
``tools/check_refine.py``; this checks the two-view wiring on top:

  1. **Passthrough wiring gate (key test):** with ``stereo_fuse`` replaced by
     a zero-output stand-in and a stereo batch whose right eye is a copy of
     the left (``T_right_ref = I``), the stereo forward must reproduce the mono
     forward's predictions bit-for-bit.  (The real fusion is ``left +
     zero-init conv(cat(left, right))`` — an exact identity at init — so the
     gate isolates the plumbing: right render, rig-composed pose, left-first
     concat order, left-fov regressor.)
  2. Random-init fusion: every prediction/loss finite at K=1..3 with the
     expected per-iteration loss keys present.
  3. Training-mode backward: finite gradients reach ``stereo_fuse`` and the
     shared backbone/regressor.
  4. Informational only: render geometrically consistent stereo observations at
     a known baseline, perturb the classification bootstrap and report the
     mono-vs-stereo depth error (fusion is untrained -> no improvement assert).
  5. Dataset smoke (skipped when no shards on disk): key presence/shapes,
     rig-transform consistency and the left->right projection identity.

Uses the fixed usprobe batch/checkpoint from ``tools/regression_guard.py``.
"""
import argparse
import dataclasses
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import pytorch3d.io  # noqa: F401,E402  (ensure pytorch3d.io submodule)
import config as cfg  # noqa: E402
import dataset_factory  # noqa: E402
import dataset_schema  # noqa: E402
import models  # noqa: E402
import rendering  # noqa: E402
import utils  # noqa: E402
import regression_guard as guard  # noqa: E402

IMAGE_SIZE = (cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE)
REG_CHANNELS = 2048  # PoseDecoder.reg_pool width; stereo_fuse in = 2x this
REFINE_LOSS_KEYS = ['quat_reg', 'txty_reg', 'tz_reg']
GEOMETRY_BASELINE = 0.063  # metres, informational only


class ZeroResidual(torch.nn.Module):
    """Stand-in for the zero-init residual ``stereo_fuse`` conv.

    The fusion is ``left + stereo_fuse(cat(left, right))``; a zero-output
    stand-in makes the fused tensor equal the left view exactly, matching
    the conv's zero initialization.
    """

    def forward(self, x):
        return torch.zeros_like(x[:, :REG_CHANNELS])


def build_stereo_model(device, load_mono=True):
    model_config = dataclasses.replace(
        cfg.ModelConfig.from_dataset_config(guard.DATASET),
        stereo_fusion=True, fusion_residual=True)
    model = models.MRCNet(model_config).to(device)
    if load_mono:
        checkpoint = torch.load(
            guard.CHECKPOINT, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(
            checkpoint['network'], strict=False)
        fuse_missing = [k for k in missing if k.startswith('stereo_fuse.')]
        assert fuse_missing, 'checkpoint unexpectedly already had stereo_fuse'
        print('stereo model strict=False: random fusion={} unexpected={}'.format(
            sorted(fuse_missing), sorted(unexpected)))
    model.eval()
    return model


def build_stereo_aux(device, right_inputs=None, T_right_ref=None):
    inputs, aux = guard.build_inputs(device)
    aux = dict(aux)
    n = aux['obj_cls'].shape[0]
    aux['inputs_right'] = (inputs.clone() if right_inputs is None
                           else right_inputs)
    aux['intrinsics_right'] = aux['intrinsics'].clone()
    aux['fov_right'] = aux['fov'].clone()
    if T_right_ref is None:
        T_right_ref = torch.eye(4, device=device).unsqueeze(0).repeat(n, 1, 1)
    aux['T_right_ref'] = T_right_ref
    return inputs, aux


def run_forward(model, inputs, aux, targets=None, n_refine_iters=1,
                grad_ckpt=False):
    with torch.no_grad():
        torch.use_deterministic_algorithms(True)
        try:
            return model(inputs, aux, targets, n_refine_iters=n_refine_iters,
                         grad_ckpt=grad_ckpt)
        except RuntimeError:
            torch.use_deterministic_algorithms(True, warn_only=True)
            return model(inputs, aux, targets, n_refine_iters=n_refine_iters,
                         grad_ckpt=grad_ckpt)


def expected_loss_keys(n_refine_iters):
    keys = list(guard.LOSS_KEYS)
    for k in range(1, n_refine_iters):
        keys.extend('iter{}/{}'.format(k, name) for name in REFINE_LOSS_KEYS)
    return keys


def check_passthrough(device):
    """Key wiring test: left-select fusion + identical views == mono forward."""
    failures = []
    mono = guard.build_model(device)

    inputs, aux_mono = guard.build_inputs(device)
    mono_preds = run_forward(mono, inputs, aux_mono)
    # free the mono model + activations before the stereo pass: the two
    # models together peak past 16 GB on this box
    mono_preds = {k: v.detach().cpu().float() for k, v in mono_preds.items()}
    del mono, inputs, aux_mono
    torch.cuda.empty_cache()

    stereo = build_stereo_model(device)
    stereo_inputs, aux_stereo = build_stereo_aux(device)
    saved_fuse = stereo.stereo_fuse
    stereo.stereo_fuse = ZeroResidual()
    try:
        stereo_preds = run_forward(stereo, stereo_inputs, aux_stereo)
    finally:
        stereo.stereo_fuse = saved_fuse
    stereo_preds = {k: v.detach().cpu().float() for k, v in stereo_preds.items()}

    for key in guard.PRED_KEYS:
        a = mono_preds[key]
        b = stereo_preds[key]
        diff = float((a - b).abs().max()) if a.numel() else 0.0
        ok = torch.allclose(a, b, rtol=1e-4, atol=1e-5)
        print('passthrough {:<14} {} max_abs_diff={:.3e}'.format(
            key, 'OK' if ok else 'FAIL', diff))
        if not ok:
            failures.append('passthrough {} diff={:.3e}'.format(key, diff))
    extra = {'render_right', 'mask_synt_right'}
    if not extra.issubset(stereo_preds):
        failures.append('stereo predictions missing debug keys {}'.format(
            sorted(extra - set(stereo_preds))))
    if extra & set(mono_preds):
        failures.append('mono predictions leaked stereo debug keys')
    return failures


def check_random_fusion(device):
    failures = []
    model = build_stereo_model(device)
    inputs, aux = build_stereo_aux(device)
    targets = guard.build_targets(device, aux)
    for k in (1, 2, 3):
        preds = run_forward(model, inputs, aux, targets, n_refine_iters=k)
        for key in guard.PRED_KEYS:
            if key not in preds:
                failures.append('K={} missing {}'.format(k, key))
            elif not torch.isfinite(preds[key].float()).all():
                failures.append('K={} non-finite {}'.format(k, key))
        for key in ('render_right', 'mask_synt_right'):
            if key not in preds:
                failures.append('K={} missing stereo debug {}'.format(k, key))
        loss_keys = expected_loss_keys(k)
        if set(preds['losses']) != set(loss_keys):
            failures.append('K={} loss keys {} != {}'.format(
                k, sorted(preds['losses']), sorted(loss_keys)))
        for key in loss_keys:
            if key in preds['losses'] and not torch.isfinite(
                    preds['losses'][key].float()).all():
                failures.append('K={} non-finite loss {}'.format(k, key))
        print('K={} stereo predictions/losses: {} ({} loss keys)'.format(
            k, 'OK' if not failures else 'FAIL', len(preds['losses'])))
    return failures


def check_backward(device):
    failures = []
    torch.use_deterministic_algorithms(False)
    model = build_stereo_model(device)
    inputs, aux = build_stereo_aux(device)
    targets = guard.build_targets(device, aux)
    model.train()
    model.zero_grad(set_to_none=True)
    preds = model(inputs, aux, targets, n_refine_iters=2)
    loss = sum(preds['losses'].values())
    if not torch.isfinite(loss):
        failures.append('backward: non-finite loss')
    else:
        loss.backward()
        n_grad, n_bad = 0, 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            n_grad += 1
            if not torch.isfinite(param.grad.float()).all():
                n_bad += 1
                failures.append('non-finite grad for {}'.format(name))
        fuse_missing = [
            n for n, p in model.stereo_fuse.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad.float()).all()]
        if fuse_missing:
            failures.append('stereo_fuse grads missing/non-finite: {}'.format(
                fuse_missing))
        for label, module in (('regressor', model.regressor),
                              ('backbone', model.backbone)):
            if not any(p.grad is not None for p in module.parameters()):
                failures.append('{} received no gradient'.format(label))
        print('backward: loss={:.4f}, {} params with grads, {} non-finite, '
              'stereo_fuse grads OK={}'.format(
                  float(loss.detach()), n_grad, n_bad, not fuse_missing))
    model.zero_grad(set_to_none=True)
    model.eval()
    return failures


def check_geometry(device):
    """Informational: stereo observations rendered at a known baseline."""
    try:
        mono = guard.build_model(device)
        stereo = build_stereo_model(device)
        _, aux = guard.build_inputs(device)
        targets = guard.build_targets(device, aux)
        obj_cls = aux['obj_cls']
        R = targets['roi_obj_R']
        t = targets['roi_obj_t']
        n = len(obj_cls)
        T = torch.eye(4, device=device).unsqueeze(0).repeat(n, 1, 1)
        T[:, :3, 3] = torch.tensor(
            [-GEOMETRY_BASELINE, 0.0, 0.0], device=device)
        R_r = T[:, :3, :3] @ R
        t_r = (T[:, :3, :3] @ t.unsqueeze(-1)).squeeze(-1) + T[:, :3, 3]
        with torch.no_grad():
            out_l = stereo.renderer.render(
                obj_cls, R, t, [rendering.CameraView(K=aux['intrinsics'])])
            out_r = stereo.renderer.render(
                obj_cls, R_r, t_r, [rendering.CameraView(K=aux['intrinsics'])])
        left_inputs = torch.cat([out_l['rgb'][:, 0], out_l['bbox_map'][:, 0]],
                                dim=1)
        right_inputs = torch.cat([out_r['rgb'][:, 0], out_r['bbox_map'][:, 0]],
                                 dim=1)
        s_inputs, s_aux = build_stereo_aux(device, right_inputs=right_inputs,
                                           T_right_ref=T)

        def z_err(preds):
            t_pred = utils.perspective_to_trans_3d(
                preds['translation'], IMAGE_SIZE, aux['intrinsics'])
            return float((t_pred[:, 2] - t[:, 2]).abs().mean())

        with torch.no_grad():
            mono_preds = mono(left_inputs, aux)
            stereo_preds = stereo(s_inputs, s_aux)
        # The usprobe weights + synthetic fov domain gap dominate this number,
        # so it is a wiring sanity print only, not an accuracy measurement.
        print('geometry (informational; usprobe weights on synthetic renders '
              '-> absolute error not meaningful): mean |z_err| mono={:.4f} m '
              'stereo={:.4f} m (baseline={:.3f} m)'.format(
                  z_err(mono_preds), z_err(stereo_preds), GEOMETRY_BASELINE))
    except Exception as exc:  # informational only; never fail the suite
        print('geometry check skipped ({}: {})'.format(
            type(exc).__name__, exc))
    return []


def check_dataset_smoke(device=None):
    """Load a few val samples; skipped when no shards are on disk."""
    failures = []
    data_dir = os.path.join(cfg.DATASET_ROOT, 'stereobj-1m')
    webds = os.path.join(data_dir, cfg.DATASET_CONFIG['stereobj-1m']['webds_dir'])
    if not os.path.isdir(webds) or not any(
            f.endswith('.tar') for f in os.listdir(webds)):
        print('SKIP dataset smoke: no shards under {}'.format(webds))
        return failures
    import numpy as np
    ds = dataset_factory.create_dataset('stereobj-1m', split='val')
    if ds.stereo_mode != 'stereo':
        failures.append("dataset stereo_mode={!r} != 'stereo'".format(
            ds.stereo_mode))
        return failures
    n = min(4, len(ds))
    assert n > 0, 'stereo val split has no both-eye instances'
    base = ds.baseline
    for i in range(n):
        s = ds[i]
        for key in dataset_schema.STEREO_KEYS:
            if key not in s:
                failures.append('sample {} missing {}'.format(i, key))
        if failures:
            break
        if s['roi_image_right'].shape != s['roi_image'].shape:
            failures.append('sample {} right/left image shape mismatch'.format(i))
        T = s['T_right_ref'].numpy()
        trans = T[:3, 3]
        if abs(float(np.linalg.norm(trans)) - base) > 1e-4:
            failures.append('sample {} |T_right_ref.t|={} != baseline {}'.format(
                i, float(np.linalg.norm(trans)), base))

        # left -> right projection identity on the val (un-augmented) split.
        fi = int(np.searchsorted(ds._offsets, i, side='right') - 1)
        off = i - ds._offsets[fi]
        row = int(ds._frame_rows[ds._frame_row_start[fi] + off]) >> 1
        label, _, _ = ds._load_frame_arrays(fi)
        obj_key = 'Object {}'.format(int(ds._i_mask[row]) - 1)
        bb_r = label['scaled_bboxes']['right'][obj_key]
        K_r = ds.eye_intrinsics('right')
        t_l = s['roi_obj_t'].numpy()
        z = max(float(t_l[2]), 1e-6)
        u_r_pred = K_r[0, 0] * (t_l[0] - base) / z + K_r[0, 2]
        # the label bboxes are canvas-referenced (the adapter subtracts
        # EYE_W when cropping); K_r is eye-local, so compare in eye-local
        # coordinates
        u_r_gt = 0.5 * (bb_r['x_min'] + bb_r['x_max']) - ds.EYE_W
        bw = max(bb_r['x_max'] - bb_r['x_min'], 1.0)
        rel = abs(u_r_pred - u_r_gt) / bw
        print('dataset sample {}: proj u_r pred={:.1f} gt={:.1f} rel={:.3f}'
              .format(i, u_r_pred, u_r_gt, rel))
        if rel > 0.1:
            failures.append('sample {} projection rel err {:.3f}'.format(i, rel))
    print('dataset smoke: {}'.format('OK' if not failures else 'FAIL'))
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available; falling back to CPU (slow renderer).')
        device = 'cpu'

    guard.setup_determinism()
    failures = []
    for check in (check_passthrough, check_random_fusion, check_backward,
                  check_geometry, check_dataset_smoke):
        failures += check(device)
        if device == 'cuda':
            # Stereo models are large and every check builds fresh ones.
            torch.cuda.empty_cache()

    if failures:
        print('\nSTEREO SMOKE FAILED:')
        for f in failures:
            print('  - {}'.format(f))
        sys.exit(1)
    print('\nStereo smoke passed.')


if __name__ == '__main__':
    main()