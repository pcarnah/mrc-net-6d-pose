"""Smoke test for the Phase-1 iterative refinement loop (``MRCNet.forward``).

Reuses the fixed synthetic batch, checkpoint and deterministic targets from
``tools/regression_guard.py``:

  * K=1 with targets: ``predictions['losses']`` must equal the captured WS0
    loss reference (bit-for-bit gate for the loss refactor).
  * K=2 / K=3: expected prediction keys present, every prediction tensor and
    every per-iteration loss finite, and the pose actually moves between
    iteration 0 and the final iteration.
  * ``grad_ckpt=True`` must not change the (no-grad) outputs.
  * A training-mode backward through the deep-supervised loss must reach the
    shared parameters with finite gradients.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import config as cfg  # noqa: E402
import utils  # noqa: E402
import regression_guard as guard  # noqa: E402

IMAGE_SIZE = (cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE)
REFINE_LOSS_KEYS = ['quat_reg', 'txty_reg', 'tz_reg']
# Rotation matrices are compared entry-wise: a trace/acos angle carries a
# ~1e-4 rad noise floor from R^T R rounding, which is larger than the early
# stop tolerance and would make this check flaky.
POSE_EPS = 1e-5


def run_forward(model, device, n_refine_iters, with_targets=False,
                grad_ckpt=False, refine_stop_thresh=0.0, det=True):
    """Deterministic forward; retries with warn-only when strict mode is
    unsupported by some op (mirrors ``regression_guard.run_forward``)."""
    inputs, aux = guard.build_inputs(device)
    targets = guard.build_targets(device, aux) if with_targets else None
    kwargs = dict(n_refine_iters=n_refine_iters, grad_ckpt=grad_ckpt,
                  refine_stop_thresh=refine_stop_thresh)
    with torch.no_grad():
        if det:
            torch.use_deterministic_algorithms(True)
            try:
                return model(inputs, aux, targets, **kwargs)
            except RuntimeError:
                torch.use_deterministic_algorithms(True, warn_only=True)
                if with_targets:
                    targets = guard.build_targets(device, aux)
                return model(inputs, aux, targets, **kwargs)
        return model(inputs, aux, targets, **kwargs)


def expected_loss_keys(n_refine_iters):
    keys = list(guard.LOSS_KEYS)
    for k in range(1, n_refine_iters):
        keys.extend('iter{}/{}'.format(k, name) for name in REFINE_LOSS_KEYS)
    return keys


def check_k1_loss(model, device):
    ref_path = guard.REF_PATH
    if not os.path.exists(ref_path):
        print('No loss reference at {}; run '
              '`python tools/regression_guard.py capture` first.'.format(
                  ref_path))
        sys.exit(2)
    ref = torch.load(ref_path, map_location='cpu', weights_only=True)
    assert 'losses' in ref, 'reference has no loss capture; re-run capture'
    preds = run_forward(model, device, 1, with_targets=True)
    failures = []
    for key in guard.LOSS_KEYS:
        new_l = preds['losses'][key].detach().cpu().float()
        ref_l = ref['losses'][key].float()
        diff = float((new_l - ref_l).abs().max())
        ok = torch.allclose(new_l, ref_l, rtol=1e-5, atol=1e-6)
        print('K=1 loss/{:<9} {} max_abs_diff={:.3e}'.format(
            key, 'OK' if ok else 'FAIL', diff))
        if not ok:
            failures.append('K=1 loss {} diff={:.3e}'.format(key, diff))
    if set(preds['losses']) != set(guard.LOSS_KEYS):
        failures.append('K=1 loss keys changed: {}'.format(
            sorted(preds['losses'])))
    return failures


def check_kN(model, device, n_refine_iters):
    failures = []
    preds = run_forward(model, device, n_refine_iters, with_targets=True)
    for key in guard.PRED_KEYS:
        if key not in preds:
            failures.append('K={} missing prediction {}'.format(
                n_refine_iters, key))
            continue
        value = preds[key]
        if not torch.isfinite(value.float()).all():
            failures.append('K={} non-finite prediction {}'.format(
                n_refine_iters, key))
    loss_keys = expected_loss_keys(n_refine_iters)
    if set(preds['losses']) != set(loss_keys):
        failures.append('K={} loss keys {} != expected {}'.format(
            n_refine_iters, sorted(preds['losses']), sorted(loss_keys)))
    for key in loss_keys:
        if key in preds['losses']:
            value = preds['losses'][key]
            if not torch.isfinite(value.float()).all():
                failures.append('K={} non-finite loss {}'.format(
                    n_refine_iters, key))
    print('K={} predictions/losses: {} ({} loss keys) {}'.format(
        n_refine_iters, 'OK' if not failures else 'FAIL',
        len(preds['losses']), 'all finite' if not failures else ''))
    return preds, failures


def pose_delta(preds_a, preds_b, device):
    """Per-sample rotation-matrix max abs entry difference and translation L2
    delta between two prediction dicts produced from the identical fixed
    batch."""
    _, aux = guard.build_inputs(device)
    R_a, R_b = preds_a['roi_obj_R'], preds_b['roi_obj_R']
    t_a = utils.perspective_to_trans_3d(
        preds_a['translation'], IMAGE_SIZE, aux['intrinsics'])
    t_b = utils.perspective_to_trans_3d(
        preds_b['translation'], IMAGE_SIZE, aux['intrinsics'])
    R_diff = (R_a - R_b).abs().amax(dim=(-2, -1))
    trans = torch.norm(t_a - t_b, dim=-1)
    return R_diff.detach().cpu(), trans.detach().cpu()


def check_pose_moves(model, device):
    failures = []
    p1 = run_forward(model, device, 1)
    p2 = run_forward(model, device, 2)
    p3 = run_forward(model, device, 3)
    for label, a, b in (('K=1 -> K=2', p1, p2), ('K=1 -> K=3', p1, p3),
                        ('K=2 -> K=3', p2, p3)):
        R_diff, trans = pose_delta(a, b, device)
        moved = int(((R_diff > POSE_EPS) | (trans > POSE_EPS)).sum())
        print('{} pose moved for {}/{} samples '
              '(max |dR|={:.3e} max |dt|={:.3e})'.format(
                  label, moved, len(R_diff), float(R_diff.max()),
                  float(trans.max())))
        if moved == 0:
            failures.append('{} did not change the pose'.format(label))
    return failures


def check_early_stop(model, device):
    """A threshold above the residual magnitude must stop after iteration 0
    and reproduce the K=1 estimate exactly; thresh=0 must be disabled."""
    failures = []
    full = run_forward(model, device, 4)
    if full.get('refine_iters_used') != 4:
        failures.append('thresh=0 should run all 4 iterations, got {}'.format(
            full.get('refine_iters_used')))
    stopped = run_forward(model, device, 4, refine_stop_thresh=1e9)
    if stopped.get('refine_iters_used') != 1:
        failures.append('huge threshold should stop after 1 iteration, '
                        'got {}'.format(stopped.get('refine_iters_used')))
    p1 = run_forward(model, device, 1)
    R_diff, trans = pose_delta(stopped, p1, device)
    matches = float(R_diff.max()) <= POSE_EPS and float(trans.max()) <= POSE_EPS
    if not matches:
        failures.append('early stop after iter 0 != K=1 pose '
                        '(max |dR|={:.3e}, max |dt|={:.3e})'.format(
                            float(R_diff.max()), float(trans.max())))
    print('early stop: thresh=0 used {} iters, huge thresh used {} iters, '
          'pose matches K=1={}'.format(
              full.get('refine_iters_used'), stopped.get('refine_iters_used'),
              matches))
    return failures


def check_grad_ckpt(model, device):
    failures = []
    plain = run_forward(model, device, 2, with_targets=True, grad_ckpt=False)
    ckpt = run_forward(model, device, 2, with_targets=True, grad_ckpt=True)
    for key in guard.PRED_KEYS:
        diff = float((plain[key].float() - ckpt[key].float()).abs().max())
        if diff > 1e-5:
            failures.append('grad_ckpt changed {} by {:.3e}'.format(key, diff))
    print('grad_ckpt vs plain (K=2, no-grad): {}'.format(
        'OK' if not failures else 'FAIL'))
    return failures


def check_backward(model, device, n_refine_iters, grad_ckpt):
    failures = []
    torch.use_deterministic_algorithms(False)
    inputs, aux = guard.build_inputs(device)
    targets = guard.build_targets(device, aux)
    model.train()
    model.zero_grad(set_to_none=True)
    preds = model(inputs, aux, targets, n_refine_iters=n_refine_iters,
                  grad_ckpt=grad_ckpt)
    loss = sum(preds['losses'].values())
    if not torch.isfinite(loss):
        failures.append('backward: non-finite loss (K={}, grad_ckpt={})'.format(
            n_refine_iters, grad_ckpt))
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
        if model.regressor.pose_reg.linear2.weight.grad is None:
            failures.append('regressor head received no gradient')
        print('backward K={} grad_ckpt={}: loss={:.4f}, {} params with '
              'grads, {} non-finite'.format(
                  n_refine_iters, grad_ckpt, float(loss.detach()),
                  n_grad, n_bad))
    model.zero_grad(set_to_none=True)
    model.eval()
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available; falling back to CPU (renderer is slow).')
        device = 'cpu'

    guard.setup_determinism()
    model = guard.build_model(device)

    failures = []
    failures += check_k1_loss(model, device)
    for k in (2, 3):
        _, f = check_kN(model, device, k)
        failures += f
    failures += check_pose_moves(model, device)
    failures += check_early_stop(model, device)
    failures += check_grad_ckpt(model, device)
    failures += check_backward(model, device, 2, grad_ckpt=False)
    failures += check_backward(model, device, 3, grad_ckpt=True)

    if failures:
        print('\nREFINEMENT SMOKE FAILED:')
        for f in failures:
            print('  - {}'.format(f))
        sys.exit(1)
    print('\nRefinement smoke passed.')


if __name__ == '__main__':
    main()
