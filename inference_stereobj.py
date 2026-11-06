"""GT-detection stereo inference on stereobj-1m (val split).

Reads instances annotated on both eyes through the ``StereobjDataset``
webdataset index. For every such (frame, instance) the left and right GT
bboxes seed matching 4-way Rz crop batches (same ``rot_index`` sequence) and
the model's stereo-fusion path predicts the pose in the left-camera frame,
written as a BOP-format CSV (``scene_id = 2*frame``). Requires
``stereo_mode='stereo'`` in ``DATASET_CONFIG`` and a checkpoint built with
``stereo_fusion=True``; a mismatched model/dataset fails loudly.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config as bop_cfg
import dataset_factory
import models
import utils
from bop_toolkit_lib import inout
from lib import data_utils as misc

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def make_warmup_batch(stereo=False):
    """Keyword args for a dummy 4-hypothesis ``inference_func`` call."""
    n = 4
    size = bop_cfg.INPUT_IMG_SIZE
    roi_rgb = np.zeros((n, 3, size, size), dtype=np.float32)
    obj_cls = np.zeros(n, dtype=np.int64)
    bbox_loc = np.tile(
        np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32), (n, 1))
    K = np.array([[400., 0., size / 2.],
                  [0., 400., size / 2.],
                  [0., 0., 1.]], dtype=np.float32)
    roi_camK = np.tile(K[None], (n, 1, 1))
    fov = np.zeros((n, 3), dtype=np.float32)
    fov[:, 2] = 0.625
    Rz = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    warmup = {'obj_cls': obj_cls, 'roi_rgb': roi_rgb, 'bbox_loc': bbox_loc,
              'roi_camK': roi_camK, 'fov': fov, 'Rz': Rz}
    if stereo:
        warmup.update({
            'roi_rgb_right': roi_rgb.copy(),
            'bbox_loc_right': bbox_loc.copy(),
            'roi_camK_right': roi_camK.copy(),
            'fov_right': fov.copy(),
            'T_right_ref': np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        })
    return warmup


def build_rz_crop_batch(view_image, view_cam_K, det_bbox, box_scale,
                        cx, cy, fx, fy, dataset_id2cls, inst_cls, n_rots=4):
    x1, y1, x2, y2 = det_bbox
    hr = (y2 - y1) / box_scale
    wr = (x2 - x1) / box_scale

    b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov \
        = [], [], [], [], [], []
    for rot_index in range(n_rots):
        rot_rad = rot_index * np.pi / 2
        bbox_center = torch.as_tensor([cx, cy], dtype=torch.float32)
        bbox_loc = np.array(
            [0.5 - wr/2, 0.5 - hr/2, 0.5 + wr/2, 0.5 + hr/2])

        Rz = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                       [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        T_rot = view_cam_K @ Rz @ np.linalg.inv(view_cam_K)
        center_hom = T_rot @ np.array(
            [*bbox_center, 1.0], dtype=np.float32)
        bbox_center = center_hom[:2]
        fov = np.array([(bbox_center[0] - cx) / fx,
                        (bbox_center[1] - cy) / fy, box_scale / fx])
        b_Rz.append(Rz)
        b_fov.append(fov)

        Tz = np.array([[1.0, 0.0, -0.5],
                       [0.0, 1.0, -0.5],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        T_loc = np.linalg.inv(Tz) @ Rz @ Tz
        bbox_loc = utils.transform_bounding_box(bbox_loc, T_loc)
        b_bbox_loc.append(bbox_loc)

        T_img2roi = misc.transform_to_local_ROIcrop(
            bbox_center=bbox_center, bbox_scale=box_scale,
            zoom_scale=bop_cfg.INPUT_IMG_SIZE)
        roi_camK = T_img2roi @ view_cam_K
        b_roi_camK.append(roi_camK)

        roi_rgb = misc.crop_resize_by_warp_affine(
            view_image.numpy(), np.array([cx, cy]), box_scale,
            bop_cfg.INPUT_IMG_SIZE, view_cam_K, rot_rad,
            interpolation='bilinear')
        roi_rgb = roi_rgb / 255.0  # 1x3xHxW
        b_roi_rgb.append(roi_rgb)
        b_obj_cls.append(inst_cls)

    b_obj_cls = np.stack(b_obj_cls, axis=0)
    b_roi_rgb = np.stack(b_roi_rgb, axis=0)
    b_bbox_loc = np.stack(b_bbox_loc, axis=0)
    b_roi_camK = np.stack(b_roi_camK, axis=0)
    b_fov = np.stack(b_fov, axis=0)
    b_Rz = np.stack(b_Rz, axis=0)
    return b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov


def inference_func(net, device, obj_cls, roi_rgb, bbox_loc, roi_camK, fov, Rz,
                   n_refine_iters=1, refine_stop_thresh=0.0,
                   roi_rgb_right=None, bbox_loc_right=None,
                   roi_camK_right=None, fov_right=None, T_right_ref=None,
                   init_pose_noise=None):
    batch_image = torch.from_numpy(roi_rgb).to(device)
    batch_obj_cls = torch.from_numpy(obj_cls).to(device)
    im_height, im_width = batch_image.shape[2:]
    batch_bbox_map = torch.stack([utils.make_roi(
        torch.as_tensor(x, dtype=torch.float32), im_height).to(device)
        for x in bbox_loc], dim=0).unsqueeze(1)
    batch_fov = torch.as_tensor(
        fov, dtype=torch.float32).to(device)
    batch_input = torch.cat([batch_image, batch_bbox_map], dim=1)
    intrinsics = torch.from_numpy(roi_camK).to(device)

    aux = {'obj_cls': batch_obj_cls,
           'fov': batch_fov,
           'intrinsics': intrinsics}
    if roi_rgb_right is not None:
        batch_image_r = torch.from_numpy(roi_rgb_right).to(device)
        batch_bbox_map_r = torch.stack([utils.make_roi(
            torch.as_tensor(x, dtype=torch.float32), im_height).to(device)
            for x in bbox_loc_right], dim=0).unsqueeze(1)
        aux['inputs_right'] = torch.cat([batch_image_r, batch_bbox_map_r],
                                        dim=1)
        aux['intrinsics_right'] = torch.from_numpy(
            roi_camK_right).to(device)
        aux['fov_right'] = torch.as_tensor(
            fov_right, dtype=torch.float32).to(device)
        aux['T_right_ref'] = torch.as_tensor(
            T_right_ref, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = net(batch_input, aux,
                          n_refine_iters=n_refine_iters,
                          refine_stop_thresh=refine_stop_thresh,
                          init_pose_noise=init_pose_noise)
        R_conf = predictions['quat_bin']
        R_index = torch.argmax(torch.amax(R_conf, dim=1))
        t_conf = predictions['depth_bin']
        t_index = torch.argmax(torch.amax(t_conf, dim=1))

        Rz_inv = np.transpose(Rz, [0, 2, 1])
        R_pred = Rz_inv @ predictions['roi_obj_R'].cpu().numpy()
        R_pred = R_pred[R_index]
        t_pred = utils.perspective_to_trans_3d(
            predictions['translation'],
            (bop_cfg.INPUT_SIZE, bop_cfg.INPUT_SIZE), intrinsics)
        t_pred = Rz_inv @ np.expand_dims(t_pred.cpu().numpy(), axis=-1)
        t_pred = t_pred[t_index, :, 0]

    return R_pred, t_pred


def _prep_worker_init_fn(*_):
    # mirror training's worker discipline: one thread per worker; drop any
    # fork-inherited tar handles (shared file offsets race across workers)
    import cv2
    import stereobj_dataset
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    stereobj_dataset._clear_tar_cache()


def _first_item(batch):
    return batch[0]


class _PrepDataset(torch.utils.data.Dataset):
    """Map over prep jobs; item ``i`` is job ``i``'s ``(payloads, stats)``."""

    def __init__(self, jobs, preparer):
        self.jobs = jobs
        self.preparer = preparer

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        return self.preparer(self.jobs[idx])


def prep_jobs(jobs, preparer, num_workers=0):
    """Yield ``(payloads, stats)`` per job, optionally prefetching the CPU
    prep (decode + Rz crop building) in DataLoader workers so it overlaps
    the GPU call. Order-preserving; ``num_workers=0`` runs the same
    preparer inline (serial path)."""
    if num_workers <= 0:
        for job in jobs:
            yield preparer(job)
        return
    loader = torch.utils.data.DataLoader(
        _PrepDataset(jobs, preparer), batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=_first_item,
        worker_init_fn=_prep_worker_init_fn)
    for item in loader:
        yield item


def _stratified_quota(population, budget):
    """Proportional (largest-remainder) allocation of ``budget`` samples across
    strata with sizes ``population``, capped at each stratum size and with a
    minimum of 1 per stratum while the budget allows. Returns an int array
    aligned with ``population`` summing to ``min(budget, total)``."""
    population = np.asarray(population, dtype=np.int64)
    total = int(population.sum())
    budget = min(int(budget), total)
    if budget >= total:
        return population.copy()
    alloc = np.minimum(population * budget // total, population)
    remainder = budget - int(alloc.sum())
    frac = population * budget % total
    for i in np.argsort(-frac):
        if remainder <= 0:
            break
        add = min(int(population[i] - alloc[i]), remainder)
        alloc[i] += add
        remainder -= add
    if budget >= len(population):
        for i in np.where(alloc == 0)[0]:
            donor = int(np.argmax(alloc))
            alloc[donor] -= 1
            alloc[i] += 1
    return alloc


def _stratified_subset(frame_rows, i_cls, i_sides, cls_names, budget, seed):
    """Stratified instance subset over the COMMON both-eye population.

    Strata are the object classes of the rows annotated on both eyes
    (``sides == 0b11``) -- the population stereo evaluates, and the same
    instances mono can evaluate on their left-eye row (same view, camera
    frame and GT key as the stereo row). The per-class quota is proportional
    (largest-remainder, min 1 per class while the budget allows) and the
    within-class draw is uniform under a seeded RNG, so mono and stereo runs
    select the identical instance set for a given ``budget``/``seed`` and
    checkpoint comparisons are apples-to-apples.

    Returns a new ``{frame_index: [(row, 'left'), ...]}`` covering only the
    selected both-eye instances.
    """
    rng = np.random.default_rng(seed)
    by_cls = {}
    for fi, items in frame_rows.items():
        for row, eye in items:
            if eye == 'left' and int(i_sides[row]) == 0b11:
                by_cls.setdefault(int(i_cls[row]), []).append((fi, row))
    classes = sorted(by_cls)
    quota = _stratified_quota([len(by_cls[c]) for c in classes], budget)
    selected = []
    for c, k in zip(classes, quota):
        pool = sorted(by_cls[c])
        selected.extend(pool[i] for i in rng.choice(
            len(pool), size=int(k), replace=False))
        print('  strat {}: {}/{} instances'.format(
            cls_names[c], int(k), len(pool)))
    subset = {}
    for fi, row in selected:
        subset.setdefault(fi, []).append((row, 'left'))
    print('Stratified subset: {}/{} both-eye instances over {} object types '
          '(seed={}; identical across mono/stereo)'.format(
              len(selected), sum(len(v) for v in by_cls.values()),
              len(classes), seed))
    return subset


def _eye_box(label, obj_key, view_eye, x_off, img_H, img_W):
    """Scaled GT box of ``obj_key`` in one eye's half-image, as a crop spec."""
    bb = label['scaled_bboxes'][view_eye][obj_key]
    x1 = bb['x_min'] - x_off
    x2 = bb['x_max'] - x_off
    y1, y2 = bb['y_min'], bb['y_max']
    cx = min((x1 + x2) / 2.0, img_W)
    cy = min((y1 + y2) / 2.0, img_H)
    bw = int(max(0, min(x2 - x1, img_W)))
    bh = int(max(0, min(y2 - y1, img_H)))
    return (x1, y1, x2, y2), \
        max(bw, bh) * bop_cfg.ZOOM_PAD_SCALE, cx, cy


class _ZeroCorr(torch.nn.Module):
    """Inference-time ablation: keep the probe's output shape but zero the
    correlation volume, so the fusion convs lose the disparity channels."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input1, input2):
        return torch.zeros_like(self.inner(input1, input2))


class StereobjFramePreparer:
    """Picklable per-frame prep for one ``(frame_index, [(row, eye)...])`` job.

    Reuses the dataset's IO layer (``_load_frame_arrays``: cached tar handles
    + decode) and builds the 4-way Rz crop batches for every instance,
    returning one payload per instance: the ``inference_func`` keyword
    arguments plus the CSV metadata. Picklable so DataLoader workers (fork or
    spawn) can run it ahead of the GPU call.
    """

    def __init__(self, dataset, stereo, n_rots, dataset_id2cls, donors=None):
        self.dataset = dataset
        self.stereo = stereo
        self.n_rots = n_rots
        self.dataset_id2cls = dataset_id2cls
        self.baseline = dataset.baseline
        self.eye_index = {'left': 0, 'right': 1}
        # --shuffle_right ablation: {row: (donor_fi, donor_row)}; the right
        # eye is cropped from the donor frame with the donor's bbox geometry
        # (T_right_ref stays: it only encodes rot_index + baseline, which are
        # instance-independent).
        self.donors = donors or {}

    def __call__(self, job):
        ds = self.dataset
        fi, items = job
        label, img_pair, _mask = ds._load_frame_arrays(fi)
        left_image = torch.as_tensor(
            np.ascontiguousarray(img_pair[:, :ds.EYE_W]),
            dtype=torch.float32)
        right_image = torch.as_tensor(
            np.ascontiguousarray(img_pair[:, ds.EYE_W:2 * ds.EYE_W]),
            dtype=torch.float32)
        img_H, img_W = left_image.shape[:2]
        K_left = ds.eye_intrinsics('left')
        K_right = ds.eye_intrinsics('right')
        fx_l, fy_l = K_left[0, 0], K_left[1, 1]
        fx_r, fy_r = K_right[0, 0], K_right[1, 1]
        im_id = int(ds._f_image_id[fi])

        payloads = []
        for row, eye in items:
            obj_key = 'Object {}'.format(int(ds._i_mask[row]) - 1)
            inst_cls = torch.as_tensor(
                int(ds._i_cls[row]), dtype=torch.int64)
            inst_name = ds.cls_names[int(ds._i_cls[row])]

            if self.stereo:
                box_l, scale_l, cx_l, cy_l = _eye_box(
                    label, obj_key, 'left', 0, img_H, img_W)
                # Right-eye ablation: crop the donor frame's right half with
                # the donor instance's bbox geometry (content + geometry from
                # another frame break true stereo correspondence).
                r_label, r_row, r_image = label, row, right_image
                if self.donors:
                    dfi, drow = self.donors.get(int(row), (fi, row))
                    if drow != int(row):
                        d_label, d_img_pair, _ = ds._load_frame_arrays(dfi)
                        r_image = torch.as_tensor(
                            np.ascontiguousarray(
                                d_img_pair[:, ds.EYE_W:2 * ds.EYE_W]),
                            dtype=torch.float32)
                        r_label, r_row = d_label, drow
                r_obj_key = 'Object {}'.format(int(ds._i_mask[r_row]) - 1)
                box_r, scale_r, cx_r, cy_r = _eye_box(
                    r_label, r_obj_key, 'right', ds.EYE_W, img_H, img_W)
                b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov = \
                    build_rz_crop_batch(
                        left_image, K_left, box_l, scale_l, cx_l, cy_l,
                        fx_l, fy_l, self.dataset_id2cls, inst_cls,
                        n_rots=self.n_rots)
                _, _, r_roi_rgb, r_bbox_loc, r_roi_camK, r_fov = \
                    build_rz_crop_batch(
                        r_image, K_right, box_r, scale_r, cx_r, cy_r,
                        fx_r, fy_r, self.dataset_id2cls, inst_cls,
                        n_rots=self.n_rots)
                T_right_ref = np.tile(
                    np.eye(4, dtype=np.float32), (b_Rz.shape[0], 1, 1))
                T_right_ref[:, :3, 3] = -np.einsum(
                    'nij,j->ni', b_Rz,
                    np.array([self.baseline, 0.0, 0.0], dtype=np.float32))
                call = dict(
                    obj_cls=b_obj_cls, roi_rgb=b_roi_rgb,
                    bbox_loc=b_bbox_loc, roi_camK=b_roi_camK, fov=b_fov,
                    Rz=b_Rz, roi_rgb_right=r_roi_rgb,
                    bbox_loc_right=r_bbox_loc, roi_camK_right=r_roi_camK,
                    fov_right=r_fov, T_right_ref=T_right_ref)
                scene_id = int(ds._f_scene_id[fi]) * 2
            else:
                x_off = ds.EYE_W if eye == 'right' else 0
                view_image = right_image if eye == 'right' else left_image
                view_K = K_right if eye == 'right' else K_left
                fx, fy = view_K[0, 0], view_K[1, 1]
                box, scale, cx, cy = _eye_box(
                    label, obj_key, eye, x_off, img_H, img_W)
                b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov = \
                    build_rz_crop_batch(
                        view_image, view_K, box, scale, cx, cy, fx, fy,
                        self.dataset_id2cls, inst_cls, n_rots=self.n_rots)
                call = dict(
                    obj_cls=b_obj_cls, roi_rgb=b_roi_rgb,
                    bbox_loc=b_bbox_loc, roi_camK=b_roi_camK, fov=b_fov,
                    Rz=b_Rz)
                scene_id = int(ds._f_scene_id[fi]) * 2 + self.eye_index[eye]
            payloads.append({'inst_name': inst_name, 'scene_id': scene_id,
                             'im_id': im_id, 'call': call})
        return payloads, {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stereobj-1m')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='webds split to run on (test shards carry no '
                             'GT boxes; only val is usable in GT mode)')
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--output_suffix', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--n_refine_iters', type=int, default=1)
    parser.add_argument('--refine_stop_thresh', type=float, default=0.0)
    parser.add_argument('--stereo_mode', type=str, default=None,
                        choices=['mono', 'stereo'],
                        help='override DATASET_CONFIG stereo_mode (default: '
                             'the dataset config value)')
    parser.add_argument('--fusion', type=str, default=None,
                        choices=list(bop_cfg.FUSION_MODES),
                        help='override the fusion operation (default: the '
                             "checkpoint's model_config)")
    parser.add_argument('--init_rot_noise_deg', type=float, default=0.0,
                        help='perturb the iteration-0 bootstrap rotation by up '
                             'to this many degrees (robustness study)')
    parser.add_argument('--init_trans_noise', type=float, default=0.0,
                        help='perturb the iteration-0 bootstrap translation by '
                             'up to this fraction of depth (robustness study)')
    parser.add_argument('--no_compile', action='store_true')
    parser.add_argument('--single_crop', action='store_true',
                        help='evaluate only the rot_index=0 crop instead of '
                             'argmax over the 4 Rz crops (use when the '
                             'classification confidence is miscalibrated)')
    parser.add_argument('--max_instances', type=int, default=0,
                        help='debug cap on the number of instances (0 = all; '
                             'with --stratified the cap is filled '
                             'proportionally across object types)')
    parser.add_argument('--stratified', action='store_true',
                        help='with --max_instances > 0, draw the subset by '
                             'stratified sampling across all object types '
                             'over the common both-eye population (mono '
                             'scores the left eye), so mono and stereo runs '
                             'select the identical instance set for a given '
                             '--seed')
    parser.add_argument('--seed', type=int, default=bop_cfg.RANDOM_SEED,
                        help='RNG seed for --stratified subset selection')
    parser.add_argument('--zero_corr', action='store_true',
                        help='inference-time ablation: zero the stereo_corr '
                             'volume so the fusion convs lose the disparity '
                             'channels')
    parser.add_argument('--shuffle_right', action='store_true',
                        help='inference-time ablation: replace each right-eye '
                             'crop with one from another frame (donor content '
                             '+ bbox geometry; breaks true stereo '
                             'correspondence)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers prefetching the CPU prep '
                             '(decode + Rz crop building) to overlap the GPU '
                             'call (0 = serial: prep on the main process)')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='override config.DATASET_ROOT (default: env '
                             'MRC_DATASET_ROOT / DATASET_ROOT, else config)')
    parser.add_argument('--eval_root', type=str, default=None,
                        help='override config.EVAL_ROOT (default: env '
                             'MRC_EVAL_ROOT / EVAL_ROOT, else DATASET_ROOT)')
    args = parser.parse_args()

    bop_cfg.set_roots(args.dataset_root, args.eval_root)
    bop_cfg.set_runtime_overrides(
        args.dataset, stereo_mode=args.stereo_mode, fusion=args.fusion)
    init_pose_noise = (args.init_rot_noise_deg, args.init_trans_noise)
    if not any(v > 0 for v in init_pose_noise):
        init_pose_noise = None

    p = {
        'dataset': args.dataset,
        'eval_root': bop_cfg.EVAL_ROOT,
        # checkpoint_name may address a subdirectory (e.g. ablation/<name>);
        # keep the output filename flat (both path separators, Windows-safe)
        'output_suffix_name': '{}_{}'.format(
            args.checkpoint_name, args.output_suffix).replace(
                '/', '_').replace('\\', '_'),
        'checkpoint': './{}/{}.pth'.format(args.checkpoint_name,
                                           args.model_name)
    }
    dataset_id2cls = bop_cfg.DATASET_CONFIG[p['dataset']]['id2cls']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(p['checkpoint'], map_location=device)
    if 'model_config' in checkpoint:
        model_config = bop_cfg.ModelConfig(**checkpoint['model_config'])
        if model_config.dataset != p['dataset']:
            print('WARNING: checkpoint dataset {} != CLI dataset {}'.format(
                model_config.dataset, p['dataset']))
    else:
        model_config = bop_cfg.ModelConfig.from_dataset_config(
            p['dataset'], stereo_mode=args.stereo_mode, fusion=args.fusion)
    net = models.MRCNet(model_config).to(device)
    print('building model for {}'.format(p['dataset']))
    print('loading pre-trained model from {}'.format(p['checkpoint']))
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
    model_stereo = bool(net.stereo_fusion)

    if args.zero_corr:
        if not getattr(net, 'fusion_corr', False):
            raise SystemExit('--zero_corr: the checkpoint has no stereo_corr '
                             'probes (fusion_corr=False)')
        for probe in list(net.stereo_corr.keys()):
            net.stereo_corr[probe] = _ZeroCorr(net.stereo_corr[probe])
        print('ablation: stereo_corr outputs zeroed')

    if args.no_compile:
        print('skipping torch.compile (--no_compile)')
    else:
        stereo_model = net.stereo_fusion
        net = torch.compile(net)
        try:
            inference_func(net, device,
                           **make_warmup_batch(stereo=stereo_model),
                           n_refine_iters=args.n_refine_iters,
                           refine_stop_thresh=0.0)
            print('torch.compile warm-up succeeded.')
        except Exception as exc:
            print('torch.compile failed on warm-up ({}); falling back to '
                  'eager.'.format(type(exc).__name__))
            net = net._orig_mod

    dataset = dataset_factory.create_dataset(p['dataset'], split=args.split)
    stereo = getattr(dataset, 'stereo_mode', 'mono') == 'stereo'
    if stereo != model_stereo:
        raise SystemExit(
            "dataset stereo_mode={!r} does not match the checkpoint's "
            "stereo_fusion={} ({}). Pass --stereo_mode to match the "
            "checkpoint.".format(
                getattr(dataset, 'stereo_mode', 'mono'), model_stereo,
                p['checkpoint']))
    est_pose_file = '{}/mrcnet_{}-{}_{}.csv'.format(
        p['eval_root'], p['dataset'], args.split, p['output_suffix_name'])

    # Group instances by frame; one webp/json load per frame serves all of
    # them.  stereo: one row per both-eye instance (left frame).  mono: one row
    # per annotated (instance, eye); scene_id encodes the eye.
    frame_rows = {}
    for row in range(len(dataset._i_frame)):
        sides = int(dataset._i_sides[row])
        fi = int(dataset._i_frame[row])
        if stereo:
            if sides == 0b11:
                frame_rows.setdefault(fi, []).append((row, 'left'))
        else:
            if sides & 1:
                frame_rows.setdefault(fi, []).append((row, 'left'))
            if sides & 2:
                frame_rows.setdefault(fi, []).append((row, 'right'))
    n_inst = sum(len(v) for v in frame_rows.values())
    print('Evaluation on {} [{}]: {} instances in {} images'.format(
        p['dataset'], 'stereo' if stereo else 'mono', n_inst, len(frame_rows)))
    print(est_pose_file)

    if args.max_instances and args.stratified:
        frame_rows = _stratified_subset(
            frame_rows, dataset._i_cls, dataset._i_sides, dataset.cls_names,
            args.max_instances, args.seed)

    donors = None
    if args.shuffle_right:
        if not stereo:
            raise SystemExit('--shuffle_right requires the stereo dataset '
                             'mode (mono has no right eye to shuffle)')
        rng_donors = np.random.default_rng(args.seed + 1)
        pool = [(fi, row) for fi, items in frame_rows.items()
                for row, eye in items]
        donors = {}
        for fi, row in pool:
            while True:
                cand = pool[rng_donors.integers(len(pool))]
                if cand[1] != row:
                    donors[row] = cand
                    break
        print('ablation: right-eye donors shuffled for {} instances '
              '(donor seed={})'.format(len(donors), args.seed + 1))

    n_rots = 1 if args.single_crop else 4
    stop_thresh = (args.refine_stop_thresh
                   if args.n_refine_iters > 1 else 0.0)
    preparer = StereobjFramePreparer(dataset, stereo, n_rots, dataset_id2cls,
                                     donors=donors)
    jobs = sorted(frame_rows.items())
    pose_results = []
    for payloads, _stats in tqdm(prep_jobs(jobs, preparer, args.num_workers),
                                 total=len(jobs), desc='images', unit='img'):
        frame_objs = []
        inst_time = []
        for payload in payloads:
            inst_timer = time.time()
            (est_R, est_t), _run_time = timed(lambda: inference_func(
                net, device, init_pose_noise=init_pose_noise,
                n_refine_iters=args.n_refine_iters,
                refine_stop_thresh=stop_thresh, **payload['call']))
            inst_time.append(time.time() - inst_timer)
            frame_objs.append((payload, est_R, est_t))
        view_cost = float(np.sum(inst_time)) if inst_time else 0.0
        for payload, est_R, est_t in frame_objs:
            pose_results.append({
                'time': view_cost,
                # scene_id encodes the frame (and eye in mono); poses are
                # expressed in the eye's camera frame.
                'scene_id': payload['scene_id'],
                'im_id': payload['im_id'],
                'obj_id': dataset_id2cls[payload['inst_name']] + 1,
                'score': 1.0,
                'R': est_R,
                't': est_t,
            })
        if args.max_instances and len(pose_results) >= args.max_instances:
            break

    inout.save_bop_results(est_pose_file, pose_results)
    print('Results saved to {}.'.format(est_pose_file))
