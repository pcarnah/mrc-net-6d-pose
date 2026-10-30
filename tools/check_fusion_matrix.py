"""Smoke-test the training path for every (stereo_mode, fusion, K) combination.

Runs real forward + backward + optimizer steps at a tiny batch size on the
stereobj-1m val split (cached index, so no shard re-scan) and asserts, for each
combination:

  * every expected loss key is present and finite (incl. ``iter{k}/...`` for
    K > 1),
  * a shared parameter receives a finite gradient,
  * the fusion op is actually selected: ``stereo_fuse`` receives a gradient
    only for ``cross_view`` and does not exist for mono.

Usage:
    python tools/check_fusion_matrix.py [--batch_size 2] [--steps 2]
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg  # noqa: E402
import dataset_factory  # noqa: E402
import models  # noqa: E402

BASE_LOSS_KEYS = ['quat_clf', 'quat_reg', 'depth_clf', 'txty_clf', 'txty_reg',
                  'tz_reg', 'mask_visib']
REFINE_LOSS_KEYS = ['quat_reg', 'txty_reg', 'tz_reg']
GPU_KEYS = ['roi_obj_R', 'roi_obj_t', 'roi_mask', 'quat_bin', 'roi_camK',
            'obj_cls', 'fov']


def expected_loss_keys(k):
    keys = list(BASE_LOSS_KEYS)
    for i in range(1, k):
        keys.extend('iter{}/{}'.format(i, n) for n in REFINE_LOSS_KEYS)
    return keys


def build_batch(batch, device):
    inputs = torch.concat(
        [batch['roi_image'], batch['bbox_map']], dim=1).to(device)
    targets = {k: batch[k].to(device) for k in GPU_KEYS}
    aux = {'obj_cls': targets['obj_cls'], 'fov': targets['fov'],
           'intrinsics': targets['roi_camK']}
    if 'roi_image_right' in batch:
        aux['inputs_right'] = torch.concat(
            [batch['roi_image_right'], batch['bbox_map_right']],
            dim=1).to(device)
        aux['intrinsics_right'] = batch['roi_camK_right'].to(device)
        aux['fov_right'] = batch['fov_right'].to(device)
        aux['T_right_ref'] = batch['T_right_ref'].to(device)
    return inputs, aux, targets


def run_combo(dataset, mode, fusion, k, batch_size, steps, device):
    mc = cfg.ModelConfig.from_dataset_config(
        'stereobj-1m', stereo_mode=mode, fusion=fusion)
    net = models.MRCNet(mc).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=True)
    it = iter(loader)
    loss_dict = None
    for _ in range(steps):
        batch = next(it)
        inputs, aux, targets = build_batch(batch, device)
        preds = net(inputs, aux, targets, n_refine_iters=k)
        loss_dict = preds['losses']
        loss = sum(loss_dict.values())
        assert torch.isfinite(loss), (mode, fusion, k, 'non-finite loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    keys = set(loss_dict.keys())
    exp = set(expected_loss_keys(k))
    assert keys == exp, (mode, fusion, k, 'loss keys differ:',
                         sorted(keys ^ exp))
    for name, v in loss_dict.items():
        assert torch.isfinite(v), (mode, fusion, k, name)

    grads = {n: p.grad for n, p in net.named_parameters()}
    assert any(g is not None and torch.isfinite(g).all()
               for g in grads.values()), (mode, fusion, k, 'no finite grads')
    fuse_grad = grads.get('stereo_fuse.conv.weight')
    if mode == 'stereo':
        if fusion == 'cross_view':
            assert fuse_grad is not None and float(fuse_grad.abs().sum()) > 0, \
                (mode, fusion, k, 'cross_view did not use stereo_fuse')
        else:
            assert fuse_grad is None, \
                (mode, fusion, k, 'stereo_fuse used for ' + str(fusion))
    else:
        assert fuse_grad is None, (mode, fusion, k, 'mono has stereo_fuse')

    return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print('device={} batch_size={} steps={} split={}'.format(
        device, args.batch_size, args.steps, args.split))

    n_ok = 0
    for mode in ('stereo', 'mono'):
        cfg.set_runtime_overrides('stereobj-1m', stereo_mode=mode)
        dataset = dataset_factory.create_dataset(
            'stereobj-1m', split=args.split)
        assert getattr(dataset, 'stereo_mode', 'mono') == mode
        fusions = list(cfg.FUSION_MODES) if mode == 'stereo' else [None]
        for fusion in fusions:
            for k in (1, 2, 3):
                loss = run_combo(dataset, mode, fusion, k, args.batch_size,
                                 args.steps, device)
                print('  OK  mode={:<6s} fusion={:<10s} K={}  loss={:.4f}'
                      .format(mode, fusion or '-', k, loss))
                n_ok += 1

    print('\nALL {} COMBINATIONS PASSED'.format(n_ok))


if __name__ == '__main__':
    main()
