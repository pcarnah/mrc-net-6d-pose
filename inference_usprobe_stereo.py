"""GT-detection stereo inference on usprobe.

Pairs each left scene ``2k`` with its matched right scene ``2k+1``, seeds the
4-way Rz crop batches from the GT bboxes on both eyes, and runs the
stereo-fusion model.  Poses are written in the **left**-camera frame as a
BOP-format CSV (``scene_id`` = left scene id, ``im_id`` = image id), so the
BOP toolkit evaluates against the left-eye GT.

Instance pairing mirrors ``usprobe_dataset.UsprobeDataset``: only left
annotations whose right counterpart has a corresponding pose (same rotation, a
pure x baseline) are evaluated; the rest are skipped and counted.

Requires ``stereo_mode='stereo'`` and a checkpoint built with
``stereo_fusion=True``; a mono checkpoint fails loudly (use
``inference_bop.py`` for mono usprobe).

Usage::

    python inference_usprobe_stereo.py \
        --checkpoint_name chkpt_usprobe --model_name <file> \
        --output_suffix stereo --n_refine_iters 1
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

import config as bop_cfg
import models
from bop_toolkit_lib import inout
from lib import file_io

from inference_bop import resolve_view_image_path
from inference_stereobj import (timed, make_warmup_batch, build_rz_crop_batch,
                                inference_func, prep_jobs)
from usprobe_dataset import (_pair_is_consistent, _orthonormalize,
                             pair_scenes)

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)


def bbox_to_xyxy(bbox_visib):
    """``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``."""
    x, y, w, h = bbox_visib
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_params(box_xyxy, img_W, img_H):
    """``(box_scale, cx, cy)`` for a GT box, matching ``inference_bop.py``."""
    x1, y1, x2, y2 = box_xyxy
    cx = min((x1 + x2) / 2.0, img_W)
    cy = min((y1 + y2) / 2.0, img_H)
    bw = int(max(0, min(x2 - x1, img_W)))
    bh = int(max(0, min(y2 - y1, img_H)))
    return max(bw, bh) * bop_cfg.ZOOM_PAD_SCALE, cx, cy


def build_right_ref(b_Rz, R_l, t_l, R_r, t_r):
    """Per-crop ``T_right_ref`` (left -> right camera frame) for the Rz crops.

    Same contract as ``UsprobeDataset.read_data_stereo``: the relative rotation
    is SVD-orthonormalized (the stored ``cam_R_m2c`` is only orthonormal to
    ~2e-4) and expressed in each crop's Rz-rotated frame.
    """
    R_rel = _orthonormalize(R_r) @ _orthonormalize(R_l).T
    n = b_Rz.shape[0]
    T = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for i in range(n):
        Rz = b_Rz[i]
        T[i, :3, :3] = Rz @ R_rel @ Rz.T
        T[i, :3, 3] = Rz @ (t_r - R_rel @ t_l)
    return T


class UsprobeScenePreparer:
    """Picklable per-scene prep for one ``(left_scene, right_scene)`` job.

    Loads the scene jsons, decodes every matched image, pairs annotations
    (mirroring ``UsprobeDataset``) and builds the 4-way Rz crop batches.
    Returns one payload per evaluated instance (the ``inference_func``
    keyword arguments plus CSV metadata) and the skip count. Picklable so
    DataLoader workers can run it ahead of the GPU call.
    """

    def __init__(self, split_dir, dataset_id2cls, n_rots):
        self.split_dir = split_dir
        self.dataset_id2cls = dataset_id2cls
        self.n_rots = n_rots

    def __call__(self, job):
        left_scene, right_scene = job
        left_dir = os.path.join(self.split_dir, left_scene)
        right_dir = os.path.join(self.split_dir, right_scene)
        if not os.path.isdir(right_dir):
            return [], {'skipped': 0}
        left_gt = file_io.load(os.path.join(left_dir, 'scene_gt.json'))
        right_gt = file_io.load(os.path.join(right_dir, 'scene_gt.json'))
        left_info = file_io.load(os.path.join(left_dir, 'scene_gt_info.json'))
        right_info = file_io.load(
            os.path.join(right_dir, 'scene_gt_info.json'))
        left_cam = file_io.load(os.path.join(left_dir, 'scene_camera.json'))
        right_cam = file_io.load(os.path.join(right_dir, 'scene_camera.json'))
        scene_id = int(left_scene)

        payloads = []
        n_skipped = 0
        for img_id_str in sorted(set(left_gt) & set(right_gt), key=int):
            img_id = int(img_id_str)
            left_image = torch.as_tensor(file_io.imread(
                resolve_view_image_path(left_dir, img_id), 'color', 'BGR'),
                dtype=torch.float32)
            right_image = torch.as_tensor(file_io.imread(
                resolve_view_image_path(right_dir, img_id), 'color', 'BGR'),
                dtype=torch.float32)
            img_H, img_W = left_image.shape[:2]
            K_l = np.asarray(left_cam[img_id_str]['cam_K'],
                             dtype=np.float32).reshape(3, 3)
            K_r = np.asarray(right_cam[img_id_str]['cam_K'],
                             dtype=np.float32).reshape(3, 3)
            fx_l, fy_l = K_l[0, 0], K_l[1, 1]
            fx_r, fy_r = K_r[0, 0], K_r[1, 1]

            used_right = set()
            for left_anno, left_bbox in zip(
                    left_gt[img_id_str], left_info[img_id_str]):
                obj_id = int(left_anno['obj_id'])
                if obj_id not in self.dataset_id2cls:
                    continue
                r_idx = None
                for cand, r_anno in enumerate(right_gt[img_id_str]):
                    if cand in used_right:
                        continue
                    if int(r_anno['obj_id']) == obj_id:
                        r_idx = cand
                        break
                if r_idx is None:
                    n_skipped += 1
                    continue
                used_right.add(r_idx)

                R_l = np.asarray(left_anno['cam_R_m2c'],
                                 dtype=np.float32).reshape(3, 3)
                t_l = np.asarray(left_anno['cam_t_m2c'], dtype=np.float32)
                r_anno = right_gt[img_id_str][r_idx]
                R_r = np.asarray(r_anno['cam_R_m2c'],
                                 dtype=np.float32).reshape(3, 3)
                t_r = np.asarray(r_anno['cam_t_m2c'], dtype=np.float32)
                if not _pair_is_consistent(R_l, t_l, R_r, t_r):
                    n_skipped += 1
                    continue

                box_l = bbox_to_xyxy(left_bbox['bbox_visib'])
                box_r = bbox_to_xyxy(
                    right_info[img_id_str][r_idx]['bbox_visib'])
                scale_l, cx_l, cy_l = box_params(box_l, img_W, img_H)
                scale_r, cx_r, cy_r = box_params(box_r, img_W, img_H)
                inst_cls = self.dataset_id2cls[obj_id]

                b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov = \
                    build_rz_crop_batch(
                        left_image, K_l, box_l, scale_l, cx_l, cy_l, fx_l,
                        fy_l, self.dataset_id2cls, inst_cls,
                        n_rots=self.n_rots)
                _, _, r_roi_rgb, r_bbox_loc, r_roi_camK, r_fov = \
                    build_rz_crop_batch(
                        right_image, K_r, box_r, scale_r, cx_r, cy_r, fx_r,
                        fy_r, self.dataset_id2cls, inst_cls,
                        n_rots=self.n_rots)
                T_right_ref = build_right_ref(b_Rz, R_l, t_l, R_r, t_r)
                payloads.append({
                    'scene_id': scene_id, 'im_id': img_id, 'obj_id': obj_id,
                    'call': dict(
                        obj_cls=b_obj_cls, roi_rgb=b_roi_rgb,
                        bbox_loc=b_bbox_loc, roi_camK=b_roi_camK, fov=b_fov,
                        Rz=b_Rz, roi_rgb_right=r_roi_rgb,
                        bbox_loc_right=r_bbox_loc, roi_camK_right=r_roi_camK,
                        fov_right=r_fov, T_right_ref=T_right_ref)})
        return payloads, {'skipped': n_skipped}


def load_model(args, device):
    checkpoint_path = os.path.join(
        args.checkpoint_name, '{}.pth'.format(args.model_name))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_config' in checkpoint:
        model_config = bop_cfg.ModelConfig(**checkpoint['model_config'])
        if model_config.dataset != args.dataset:
            print('WARNING: checkpoint dataset {} != CLI dataset {}'.format(
                model_config.dataset, args.dataset))
    else:
        model_config = bop_cfg.ModelConfig.from_dataset_config(
            args.dataset, stereo_mode='stereo')
    net = models.MRCNet(model_config).to(device)
    print('building model for {}'.format(args.dataset))
    print('loading pre-trained model from {}'.format(checkpoint_path))
    network = checkpoint['network']
    has_fuse = any(k.startswith('stereo_fuse.') for k in network)
    if model_config.stereo_fusion != has_fuse:
        print('WARNING: checkpoint stereo_fuse={} does not match model '
              'stereo_fusion={}; loading non-strict (fusion is random)'.format(
                  has_fuse, model_config.stereo_fusion))
        net.load_state_dict(network, strict=False)
    else:
        net.load_state_dict(network)
    net.eval()
    if not net.stereo_fusion:
        raise SystemExit(
            'checkpoint is mono (stereo_fusion=False); use inference_bop.py '
            'for mono usprobe evaluation.')
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default='usprobe')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'train'])
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--output_suffix', type=str, default='')
    parser.add_argument('--n_refine_iters', type=int, default=1)
    parser.add_argument('--refine_stop_thresh', type=float, default=0.0)
    parser.add_argument('--no_compile', action='store_true')
    parser.add_argument('--single_crop', action='store_true',
                        help='use only the rot_index=0 crop instead of the '
                             'argmax over the 4 Rz crops')
    parser.add_argument('--max_instances', type=int, default=0,
                        help='debug cap on the number of instances (0 = all; '
                             'checked per scene when prefetching)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers prefetching the CPU prep '
                             '(image decode + crop building) to overlap the '
                             'GPU call (0 = serial: prep on the main process)')
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--eval_root', type=str, default=None)
    args = parser.parse_args()

    bop_cfg.set_roots(args.dataset_root, args.eval_root)
    bop_cfg.set_runtime_overrides(args.dataset, stereo_mode='stereo')

    dataset_id2cls = bop_cfg.DATASET_CONFIG[args.dataset]['id2cls']
    split_dir = os.path.join(
        bop_cfg.DATASET_ROOT, args.dataset, args.split)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(args, device)

    if args.no_compile:
        print('skipping torch.compile (--no_compile)')
    else:
        net = torch.compile(net)
        try:
            inference_func(net, device, **make_warmup_batch(stereo=True),
                           n_refine_iters=args.n_refine_iters,
                           refine_stop_thresh=0.0)
            print('torch.compile warm-up succeeded.')
        except Exception as exc:
            print('torch.compile failed on warm-up ({}); falling back to '
                  'eager.'.format(type(exc).__name__))
            net = net._orig_mod

    est_pose_file = '{}/mrcnet_{}-{}_{}.csv'.format(
        bop_cfg.EVAL_ROOT, args.dataset, args.split,
        '{}_{}'.format(os.path.basename(args.checkpoint_name.rstrip('/')),
                       args.output_suffix))
    os.makedirs(bop_cfg.EVAL_ROOT, exist_ok=True)

    scenes = sorted(s for s in os.listdir(split_dir) if s.startswith('00'))
    pairs, leftover = pair_scenes(scenes)
    if leftover is not None:
        print('WARNING: unpaired trailing scene {}'.format(leftover))

    n_rots = 1 if args.single_crop else 4
    stop_thresh = (args.refine_stop_thresh if args.n_refine_iters > 1 else 0.0)
    pose_results = []
    n_inst = 0
    n_skipped = 0

    preparer = UsprobeScenePreparer(split_dir, dataset_id2cls, n_rots)
    for payloads, stats in tqdm(prep_jobs(pairs, preparer, args.num_workers),
                                total=len(pairs), desc='scenes', unit='scene'):
        n_skipped += stats['skipped']
        for payload in payloads:
            (est_R, est_t), _ = timed(lambda: inference_func(
                net, device, n_refine_iters=args.n_refine_iters,
                refine_stop_thresh=stop_thresh, **payload['call']))
            pose_results.append({
                'time': 0.0,
                'scene_id': payload['scene_id'],
                'im_id': payload['im_id'],
                'obj_id': payload['obj_id'],
                'score': 1.0,
                'R': est_R,
                't': est_t,
            })
            n_inst += 1
        if args.max_instances and n_inst >= args.max_instances:
            break

    inout.save_bop_results(est_pose_file, pose_results)
    print('Evaluated {} instances ({} skipped as non-corresponding).'.format(
        n_inst, n_skipped))
    print('Results saved to {}.'.format(est_pose_file))


if __name__ == '__main__':
    main()
