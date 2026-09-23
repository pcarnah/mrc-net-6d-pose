"""GT-detection inference on stereobj-1m (val split).

Mirrors ``inference_bop.py`` but reads instances through the
``StereobjDataset`` webdataset index: for every annotated (frame, eye,
instance) the GT bbox seeds the 4-way Rz crop batch and the predicted
pose is written as a BOP-format CSV (scene_id encodes frame*2+eye, so
poses stay attributable per eye for mono-mode evaluation).
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


def make_warmup_batch():
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
    return obj_cls, roi_rgb, bbox_loc, roi_camK, fov, Rz


def build_rz_crop_batch(view_image, view_cam_K, det_bbox, box_scale,
                        cx, cy, fx, fy, dataset_id2cls, inst_cls):
    x1, y1, x2, y2 = det_bbox
    hr = (y2 - y1) / box_scale
    wr = (x2 - x1) / box_scale

    b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov \
        = [], [], [], [], [], []
    for rot_index in range(4):
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
                   n_refine_iters=1, refine_stop_thresh=0.0):
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

    with torch.no_grad():
        predictions = net(batch_input,
                          {'obj_cls': batch_obj_cls,
                           'fov': batch_fov,
                           'intrinsics': intrinsics},
                          n_refine_iters=n_refine_iters,
                          refine_stop_thresh=refine_stop_thresh)
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
    parser.add_argument('--no_compile', action='store_true')
    parser.add_argument('--max_instances', type=int, default=0,
                        help='debug cap on the number of instances (0 = all)')
    args = parser.parse_args()

    p = {
        'dataset': args.dataset,
        'eval_root': bop_cfg.EVAL_ROOT,
        'output_suffix_name': '{}_{}'.format(
            args.checkpoint_name, args.output_suffix),
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
        model_config = bop_cfg.ModelConfig.from_dataset_config(p['dataset'])
    net = models.MRCNet(model_config).to(device)
    print('building model for {}'.format(p['dataset']))
    print('loading pre-trained model from {}'.format(p['checkpoint']))
    net.load_state_dict(checkpoint['network'])
    net.eval()

    if args.no_compile:
        print('skipping torch.compile (--no_compile)')
    else:
        net = torch.compile(net)
        try:
            inference_func(net, device, *make_warmup_batch(),
                           n_refine_iters=args.n_refine_iters,
                           refine_stop_thresh=0.0)
            print('torch.compile warm-up succeeded.')
        except Exception as exc:
            print('torch.compile failed on warm-up ({}); falling back to '
                  'eager.'.format(type(exc).__name__))
            net = net._orig_mod

    dataset = dataset_factory.create_dataset(p['dataset'], split=args.split)
    est_pose_file = '{}/mrcnet_{}-{}_{}.csv'.format(
        p['eval_root'], p['dataset'], args.split, p['output_suffix_name'])

    # Group the flat index by (frame, eye): one image load serves all its
    # instances.
    image_groups = {}
    for idx in range(len(dataset)):
        fi = int(np.searchsorted(dataset._offsets, idx, side='right') - 1)
        off = idx - dataset._offsets[fi]
        frame = dataset.frame_records[fi]
        seen = 0
        for inst in frame['instances']:
            for eye_i, eye in enumerate(dataset.EYE_KEYS):
                if inst['sides'] & (1 << eye_i):
                    if seen == off:
                        key = (fi, eye)
                        image_groups.setdefault(key, []).append(
                            (idx, inst, eye))
                    seen += 1

    print('Evaluation on {}: {} instances in {} images'.format(
        p['dataset'], len(dataset), len(image_groups)))
    print(est_pose_file)

    pose_results = []
    eval_steps = 0
    view_runtime = []
    for (fi, eye), entries in tqdm(sorted(image_groups.items()),
                                   desc='images', unit='img'):
        frame = dataset.frame_records[fi]
        x_off = dataset.EYE_W if eye == 'right' else 0
        import tarfile
        with tarfile.open(frame['shard'], 'r') as tar:
            webp_buf = dataset._read_member(
                tar, frame['webp_offset'], frame['webp_size'])
        import cv2
        img_pair = cv2.imdecode(
            np.frombuffer(webp_buf, np.uint8), cv2.IMREAD_COLOR)
        view_image = torch.as_tensor(
            np.ascontiguousarray(img_pair[:, x_off:x_off + dataset.EYE_W]),
            dtype=torch.float32)
        img_H, img_W = view_image.shape[:2]
        view_cam_K = dataset.eye_intrinsics(eye)
        fx, fy = view_cam_K[0, 0], view_cam_K[1, 1]
        cx0, cy0 = view_cam_K[0, 2], view_cam_K[1, 2]

        inst_time = []
        view_objs = []
        for idx, inst, _ in entries:
            import json
            with tarfile.open(frame['shard'], 'r') as tar:
                label = json.loads(dataset._read_member(
                    tar, frame['json_offset'],
                    frame['json_size']).decode('utf-8'))
            bb = label['scaled_bboxes'][eye][inst['obj_key']]
            x1 = bb['x_min'] - x_off
            x2 = bb['x_max'] - x_off
            y1, y2 = bb['y_min'], bb['y_max']
            cx = min((x1 + x2) / 2.0, img_W)
            cy = min((y1 + y2) / 2.0, img_H)
            bw = int(max(0, min(x2 - x1, img_W)))
            bh = int(max(0, min(y2 - y1, img_H)))
            box_scale = max(bw, bh) * bop_cfg.ZOOM_PAD_SCALE

            inst_cls = torch.as_tensor(
                dataset_id2cls[inst['obj_name']], dtype=torch.int64)
            inst_timer = time.time()
            b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov = \
                build_rz_crop_batch(view_image, view_cam_K,
                                    (x1, y1, x2, y2), box_scale, cx, cy,
                                    fx, fy, dataset_id2cls, inst_cls)
            stop_thresh = (args.refine_stop_thresh
                           if args.n_refine_iters > 1 else 0.0)
            (est_R, est_t), run_time = timed(lambda: inference_func(
                net, device, b_obj_cls, b_roi_rgb, b_bbox_loc,
                b_roi_camK, b_fov, b_Rz,
                n_refine_iters=args.n_refine_iters,
                refine_stop_thresh=stop_thresh))
            inst_time.append(time.time() - inst_timer)
            view_objs.append((inst, est_R, est_t))

        view_cost = np.sum(inst_time) if inst_time else 0.0
        view_runtime.append(view_cost)
        for inst, est_R, est_t in view_objs:
            pose_results.append({
                'time': view_cost,
                # BOP im_id: frame index within the split; scene_id also
                # encodes the eye (2*scene + eye) for mono-mode eval.
                'scene_id': int(frame['scene_id']) * 2 + dataset.EYE_KEYS.index(eye),
                'im_id': int(frame['key'].rsplit('_', 1)[1]),
                'obj_id': dataset_id2cls[inst['obj_name']] + 1,
                'score': 1.0,
                'R': est_R,
                't': est_t,
            })
        eval_steps += 1
        if args.max_instances and len(pose_results) >= args.max_instances:
            break

    inout.save_bop_results(est_pose_file, pose_results)
    print('Results saved to {}.'.format(est_pose_file))
