"""ADD(-S) / 2D-projection evaluation for stereobj-1m (and any BOP-format CSV).

Consumes the BOP-format CSV written by ``inference_stereobj.py`` and scores it
against the ground truth reconstructed from the ``StereobjDataset`` index
(no shard re-reads).  Supports both the stereo split (one row per both-eye
instance) and the mono split (one row per annotated eye), matching the
``scene_id`` convention used by the inference script:

    stereo : scene_id = 2 * scene_id
    mono   : scene_id = 2 * scene_id + eye_index   (left=0, right=1)

Reported per run:
  * ADD(-S): ADD for asymmetric objects, ADD-S for objects with declared
    symmetries (see ``--symmetries``); threshold 0.1 * diameter.
  * 2D projection error (mean vertex reprojection distance, px); threshold 5 px.
  * ADD(-S) AUC over [0, 0.1 * diameter].
  * Overall + per-object + per-attribute (transparent/reflective/symmetric)
    breakdown, with bootstrap CIs over test frames.

Symmetries are optional: without a ``symmetries.json`` (or ``--symmetries``)
every object is treated as asymmetric and ADD-S degenerates to ADD.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg  # noqa: E402
import dataset_factory  # noqa: E402
from bop_toolkit_lib import inout  # noqa: E402
from bop_toolkit_lib import pose_error  # noqa: E402

# Per-object semantic attributes for the per-object breakdown.  Override with
# --attrs (JSON: {name: [tags...]}).  Names not listed get no tags.
DEFAULT_ATTRS = {
    'centrifuge_tube': ['transparent', 'symmetric'],
    'microplate': ['transparent'],
    'pipette_0.5_10': ['transparent'],
    'pipette_10_100': ['transparent'],
    'pipette_100_1000': ['transparent'],
    'sterile_tip_rack_10': ['transparent'],
    'sterile_tip_rack_200': ['transparent'],
    'sterile_tip_rack_1000': ['transparent'],
    'tube_rack_1.5_2_ml': ['transparent'],
    'tube_rack_50_ml': ['transparent'],
    'blade_razor': ['reflective'],
    'hammer': ['reflective'],
    'needle_nose_pliers': ['reflective'],
    'screwdriver': ['reflective', 'symmetric'],
    'side_cutters': ['reflective'],
    'tape_measure': ['reflective', 'symmetric'],
    'wire_stripper': ['reflective'],
    'wrench': ['reflective'],
}

ADD_THRESH_FRAC = 0.1   # ADD(-S) correctness threshold, fraction of diameter
PROJ_THRESH_PX = 5.0    # 2D projection correctness threshold, pixels


def build_gt(ds, stereo):
    """Reconstruct the GT table from the dataset index.

    Returns ``{ (scene_id, im_id, obj_id): dict(R, t, cls, K) }``.
    """
    gt = {}
    baseline = ds.baseline
    f_scene = ds._f_scene_id
    f_img = ds._f_image_id
    K_left = ds.eye_intrinsics('left')
    K_right = ds.eye_intrinsics('right')
    for row in range(len(ds._i_frame)):
        sides = int(ds._i_sides[row])
        fi = int(ds._i_frame[row])
        cls = int(ds._i_cls[row])
        obj_id = cls + 1
        R = np.asarray(ds._i_R[row], dtype=np.float64)
        t = np.asarray(ds._i_t[row], dtype=np.float64)
        if stereo:
            if sides != 0b11:
                continue
            key = (int(f_scene[fi]) * 2, int(f_img[fi]), obj_id)
            gt[key] = {'R': R, 't': t, 'cls': cls, 'K': K_left}
        else:
            if sides & 1:
                key = (int(f_scene[fi]) * 2, int(f_img[fi]), obj_id)
                gt[key] = {'R': R, 't': t, 'cls': cls, 'K': K_left}
            if sides & 2:
                key = (int(f_scene[fi]) * 2 + 1, int(f_img[fi]), obj_id)
                gt[key] = {'R': R,
                           't': t - np.array([baseline, 0.0, 0.0]),
                           'cls': cls, 'K': K_right}
    return gt


def load_symmetries(dataset_name, override_path):
    """name -> list of 4x4 symmetry transforms (identity always included)."""
    path = override_path
    if path is None:
        cand = os.path.join(cfg.DATASET_ROOT, dataset_name, 'symmetries.json')
        if os.path.exists(cand):
            path = cand
    syms = {}
    if path and os.path.exists(path):
        with open(path, 'r') as fp:
            raw = json.load(fp)
        for name, mats in raw.items():
            syms[name] = [np.asarray(m, dtype=np.float64).reshape(4, 4)
                          for m in mats]
    return syms


def load_attrs(path):
    if path and os.path.exists(path):
        with open(path, 'r') as fp:
            return {k: list(v) for k, v in json.load(fp).items()}
    return dict(DEFAULT_ATTRS)


def add_s(R_est, t_est, R_gt, t_gt, pts, syms):
    """Symmetry-aware ADD (ADD-S): mean over points of the min distance to any
    symmetry-transformed GT point.  ``syms`` includes the identity."""
    pts_est = (R_est @ pts.T).T + t_est
    best = None
    for T in syms:
        R_s, t_s = T[:3, :3], T[:3, 3]
        pts_gt = (R_gt @ (R_s @ pts.T + t_s[:, None])).T + t_gt
        d = np.linalg.norm(pts_est - pts_gt, axis=1)
        best = d if best is None else np.minimum(best, d)
    return float(best.mean())


def add_auc(errors, diameter):
    """Normalized area under the recall-vs-threshold curve over [0, 0.1*d]."""
    max_t = ADD_THRESH_FRAC * diameter
    if max_t <= 0 or len(errors) == 0:
        return 0.0
    ts = np.linspace(0.0, max_t, 100)
    recalls = np.array([np.mean(errors < t) for t in ts])
    return float(np.trapezoid(recalls, ts) / max_t)


def subsample(pts, max_points):
    if max_points and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        return pts[idx]
    return pts


def score_records(records):
    """Aggregate a list of per-instance records into metric dicts."""
    if not records:
        return {}
    add_err = np.array([r['add_s'] for r in records])
    proj_err = np.array([r['proj'] for r in records])
    diam = np.array([r['diameter'] for r in records])
    add_ok = add_err < ADD_THRESH_FRAC * diam
    proj_ok = proj_err < PROJ_THRESH_PX
    return {
        'n': int(len(records)),
        'add_s_mean': float(add_err.mean()),
        'add_s_median': float(np.median(add_err)),
        'add_s_auc': float(np.mean([add_auc(
            add_err[diam == d], d) for d in np.unique(diam)])),
        'add_s_acc': float(add_ok.mean()),
        'proj_mean': float(proj_err.mean()),
        'proj_median': float(np.median(proj_err)),
        'proj_acc': float(proj_ok.mean()),
    }


def bootstrap_ci(records, n_boot, seed=0):
    """Percentile CIs over test frames (resample frames with replacement)."""
    if not records:
        return {}
    rng = np.random.default_rng(seed)
    by_frame = {}
    for r in records:
        by_frame.setdefault(r['frame'], []).append(r)
    uniq = list(by_frame.keys())
    stats = {'add_s_mean': [], 'add_s_acc': [], 'proj_mean': [],
             'proj_acc': [], 'add_s_auc': []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sample = [r for i in pick for r in by_frame[uniq[i]]]
        agg = score_records(sample)
        for k in stats:
            stats[k].append(agg[k])
    ci = {}
    for k, vals in stats.items():
        vals = np.asarray(vals)
        ci[k] = [float(np.percentile(vals, 2.5)),
                 float(np.percentile(vals, 97.5))]
    return ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True,
                        help='BOP-format CSV from inference_stereobj.py')
    parser.add_argument('--dataset', type=str, default='stereobj-1m')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'])
    parser.add_argument('--stereo_mode', type=str, default=None,
                        choices=['mono', 'stereo'])
    parser.add_argument('--output', type=str, default=None,
                        help='optional JSON path for the full report')
    parser.add_argument('--symmetries', type=str, default=None,
                        help='optional JSON {name: [4x4, ...]}')
    parser.add_argument('--attrs', type=str, default=None,
                        help='optional JSON {name: [tags...]}')
    parser.add_argument('--max_points', type=int, default=1000,
                        help='subsample mesh vertices for ADD(-S) (0 = all)')
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='override config.DATASET_ROOT (default: env '
                             'MRC_DATASET_ROOT / DATASET_ROOT, else config)')
    args = parser.parse_args()

    cfg.set_roots(args.dataset_root)
    cfg.set_runtime_overrides(args.dataset, stereo_mode=args.stereo_mode)
    ds_cfg = cfg.DATASET_CONFIG[args.dataset]
    stereo = ds_cfg.get('stereo_mode', 'mono') == 'stereo'

    ds = dataset_factory.create_dataset(args.dataset, split=args.split)
    stereo = getattr(ds, 'stereo_mode', 'mono') == 'stereo'
    gt = build_gt(ds, stereo)
    syms = load_symmetries(args.dataset, args.symmetries)
    attrs = load_attrs(args.attrs)
    registry = ds.registry
    cls_names = ds.cls_names

    results = inout.load_bop_results(args.results)
    print('{}: {} GT instances, {} estimates'.format(
        args.results, len(gt), len(results)))

    records = []
    missing = 0
    for res in results:
        key = (int(res['scene_id']), int(res['im_id']), int(res['obj_id']))
        if key not in gt:
            missing += 1
            continue
        g = gt[key]
        cls = g['cls']
        name = cls_names[cls]
        pts = subsample(
            registry.get_vertices(cls).cpu().numpy().astype(np.float64),
            args.max_points)
        diameter = float(registry.diameter[cls])
        sym = syms.get(name, [np.eye(4)])
        if not any(np.allclose(s, np.eye(4)) for s in sym):
            sym = [np.eye(4)] + list(sym)
        R_est = np.asarray(res['R'], dtype=np.float64)
        t_est = np.asarray(res['t'], dtype=np.float64).reshape(3)
        R_gt = g['R']
        t_gt = g['t']
        add_err = add_s(R_est, t_est, R_gt, t_gt, pts, sym)
        proj_err = pose_error.proj(
            R_est, t_est.reshape(3, 1), R_gt, t_gt.reshape(3, 1), g['K'], pts)
        records.append({
            'frame': (key[0], key[1]),
            'obj': name,
            'cls': cls,
            'diameter': diameter,
            'add_s': add_err,
            'proj': proj_err,
            'time': float(res.get('time', 0.0)),
            'tags': attrs.get(name, []),
        })

    if missing:
        print('WARNING: {} estimates had no matching GT (skipped)'.format(
            missing))
    covered = len({r['frame'] for r in records})
    print('Scored {} instances over {} frames.'.format(
        len(records), covered))

    report = {
        'results': args.results,
        'dataset': args.dataset,
        'split': args.split,
        'stereo_mode': 'stereo' if stereo else 'mono',
        'overall': score_records(records),
        'overall_ci95': bootstrap_ci(records, args.n_bootstrap, args.seed),
        'latency_mean_s': float(np.mean([r['time'] for r in records]))
        if records else 0.0,
    }

    # Per-object breakdown.
    by_obj = {}
    for r in records:
        by_obj.setdefault(r['obj'], []).append(r)
    report['per_object'] = {
        name: score_records(rs) for name, rs in sorted(by_obj.items())}

    # Per-attribute breakdown.
    by_attr = {}
    for r in records:
        for tag in r['tags']:
            by_attr.setdefault(tag, []).append(r)
    report['per_attribute'] = {
        tag: score_records(rs) for tag, rs in sorted(by_attr.items())}

    def fmt(m):
        if not m:
            return 'n=0'
        return ('n={n:<5d} ADD-S {add_s_mean:8.2f}mm AUC {add_s_auc:.3f} '
                'Acc {add_s_acc:.3f} | Proj {proj_mean:6.2f}px Acc '
                '{proj_acc:.3f}').format(**{k: v for k, v in m.items()})

    print('\n== Overall ==')
    print(fmt(report['overall']))
    ci = report['overall_ci95']
    if ci:
        print('  95% CI (frames): ADD-S [{:.2f}, {:.2f}]mm  Acc [{:.3f}, '
              '{:.3f}]  Proj [{:.2f}, {:.2f}]px'.format(
                  ci['add_s_mean'][0], ci['add_s_mean'][1],
                  ci['add_s_acc'][0], ci['add_s_acc'][1],
                  ci['proj_mean'][0], ci['proj_mean'][1]))
    print('\n== Per object ==')
    for name, m in report['per_object'].items():
        print('  {:<22s} {}'.format(name, fmt(m)))
    if report['per_attribute']:
        print('\n== Per attribute ==')
        for tag, m in report['per_attribute'].items():
            print('  {:<12s} {}'.format(tag, fmt(m)))

    if args.output:
        with open(args.output, 'w') as fp:
            json.dump(report, fp, indent=2)
        print('\nReport written to {}.'.format(args.output))


if __name__ == '__main__':
    main()
