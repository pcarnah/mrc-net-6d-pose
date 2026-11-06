"""Summary tables for the matched-subset ablation eval.

Table 1 macro-averages the ADD(-S) / 2D-projection metrics over objects
(unweighted per-object mean) from the per-object breakdowns in the eval JSONs
written by ``tools/eval_stereobj.py``, excluding one object (default
``centrifuge_tube``).

Table 2 decomposes the translation error into per-axis components, recomputed
from the BOP CSVs against the dataset GT (``build_gt`` with the stereo
convention covers the mono-left rows too: same keys, same GT entries):
``|dt|`` Euclidean norm, per-axis absolute means ``|dx| |dy| |dz|`` (mm), and
the signed ``dz`` bias (positive = predicted farther than GT). Same macro
protocol (per-object mean, then unweighted mean over the kept objects).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import config as cfg  # noqa: E402
import dataset_factory  # noqa: E402
from bop_toolkit_lib import inout  # noqa: E402
import eval_stereobj as es  # noqa: E402

# (display name, run_tag, epoch) in reporting order.
ARMS = [
    ('mono_K1_e2 (base)', 'mono_K1', 'epoch_002'),
    ('uf_left', 'uf_left_k1', 'epoch_000'),
    ('uf_mean', 'uf_mean_k1', 'epoch_000'),
    ('uf_reg', 'uf_reg_k1', 'epoch_000'),
    ('uf_regclf', 'uf_regclf_k1', 'epoch_000'),
    ('uf_regcorr', 'uf_regcorr_k1', 'epoch_000'),
    ('uf_full', 'uf_full_k1', 'epoch_000'),
    ('fr_full', 'fr_full_k1', 'epoch_000'),
    ('fr_reg', 'fr_reg_k1', 'epoch_000'),
    ('fr_regclf', 'fr_regclf_k1', 'epoch_000'),
    ('fr_regcorr', 'fr_regcorr_k1', 'epoch_000'),
]


def macro_from_report(rep, exclude):
    """Unweighted mean of the per-object metrics, minus ``exclude``."""
    per = {k: v for k, v in rep['per_object'].items() if k != exclude}
    if not per:
        return None
    keys = ('add_s_mean', 'add_s_auc', 'add_s_acc', 'proj_mean', 'proj_acc')
    return {k: float(np.mean([v[k] for v in per.values()])) for k in keys}


def translation_records(csv_path, gt, cls_names):
    """Per-instance translation errors (mm) matched against GT."""
    recs = []
    missing = 0
    for res in inout.load_bop_results(csv_path):
        key = (int(res['scene_id']), int(res['im_id']), int(res['obj_id']))
        g = gt.get(key)
        if g is None:
            missing += 1
            continue
        t_est = np.asarray(res['t'], dtype=np.float64).reshape(3)
        d = t_est - g['t']
        recs.append({'obj': cls_names[g['cls']],
                     'dt': float(np.linalg.norm(d)),
                     'dx': abs(float(d[0])), 'dy': abs(float(d[1])),
                     'dz': abs(float(d[2])), 'dz_signed': float(d[2])})
    if missing:
        print('WARNING: {}: {} estimates had no matching GT'.format(
            os.path.basename(csv_path), missing))
    return recs


def macro_translation(recs, exclude):
    """Per-object means, then unweighted mean over the kept objects."""
    by_obj = {}
    for r in recs:
        if r['obj'] != exclude:
            by_obj.setdefault(r['obj'], []).append(r)
    if not by_obj:
        return None
    keys = ('dt', 'dx', 'dy', 'dz', 'dz_signed')
    per_obj = {o: {k: float(np.mean([r[k] for r in rs])) for k in keys}
               for o, rs in by_obj.items()}
    return {k: float(np.mean([m[k] for m in per_obj.values()]))
            for k in keys}, sum(len(rs) for rs in by_obj.values())


def fmt(v, nd=2):
    return '{:.{}f}'.format(v, nd) if v is not None else '-'


def _resolve_tags(reports_dir, dataset, split, run_tag, epoch, suffix):
    """JSON/CSV paths for one arm. Tries the epoch-less tag first (host-run
    naming, e.g. ``uf_reg_n10000_strat_multi``), then the epoch-tagged one
    (container naming, e.g. ``uf_reg_epoch_000_n10000_strat_multi``)."""
    tags = ['{}_{}'.format(run_tag, suffix),
            '{}_{}_{}'.format(run_tag, epoch, suffix)]
    json_path = None
    for tag in tags:
        p = os.path.join(reports_dir, tag + '.json')
        if os.path.exists(p):
            json_path = p
            break
    csv_path = None
    for tag in tags:
        cands = sorted(glob.glob(os.path.join(
            reports_dir,
            'mrcnet_{}-{}_{}_{}.csv'.format(dataset, split, '*', tag))))
        if cands:
            csv_path = cands[0]
            break
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports_dir', type=str,
                        default=os.path.join(ROOT, 'reports', 'ablation_2500'))
    parser.add_argument('--dataset', type=str, default='stereobj-1m')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--exclude', type=str, default='centrifuge_tube',
                        help='object name dropped from the macro average')
    parser.add_argument('--suffix', type=str, default='n2500_strat',
                        help='output tag suffix shared by all arms')
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--output', type=str, default=None,
                        help='optional path to also write the markdown tables')
    args = parser.parse_args()

    # ---- Table 1: macro ADD(-S) / proj from the JSON reports.
    rows1, missing = [], []
    for name, run_tag, epoch in ARMS:
        json_path, _ = _resolve_tags(
            args.reports_dir, args.dataset, args.split, run_tag, epoch,
            args.suffix)
        if json_path is None:
            missing.append(name)
            rows1.append((name, None))
            continue
        with open(json_path) as fp:
            rows1.append((name, macro_from_report(json.load(fp), args.exclude)))

    # ---- Table 2: translation decomposition from the CSVs.
    cfg.set_roots(args.dataset_root)
    cfg.set_runtime_overrides(args.dataset, stereo_mode='stereo')
    ds = dataset_factory.create_dataset(args.dataset, split=args.split)
    gt = es.build_gt(ds, stereo=True)
    cls_names = ds.cls_names
    rows2 = []
    for name, run_tag, epoch in ARMS:
        _, csv_path = _resolve_tags(
            args.reports_dir, args.dataset, args.split, run_tag, epoch,
            args.suffix)
        if csv_path is None:
            rows2.append((name, None, 0))
            continue
        recs = translation_records(csv_path, gt, cls_names)
        macro, n = macro_translation(recs, args.exclude)
        rows2.append((name, macro, n))

    # ---- Markdown output.
    out = []
    out.append('Macro-averaged over objects (unweighted per-object mean), '
               'excluding **{}**; matched subset, {} GT instances per '
               'arm.'.format(args.exclude, rows2[0][2] if rows2[0][2] else '?'))
    out.append('')
    out.append('### ADD(-S) / 2D projection')
    out.append('')
    out.append('| arm | ADD-S mean (mm) | ADD-S AUC | ADD-S acc | '
               'proj mean (px) | proj acc |')
    out.append('|---|---|---|---|---|---|')
    for name, m in rows1:
        if m is None:
            out.append('| {} | - | - | - | - | - |'.format(name))
            continue
        out.append('| {} | {} | {} | {} | {} | {} |'.format(
            name, fmt(m['add_s_mean']), fmt(m['add_s_auc'], 3),
            fmt(m['add_s_acc'], 3), fmt(m['proj_mean']),
            fmt(m['proj_acc'], 3)))
    out.append('')
    out.append('### Translation error decomposition (mm)')
    out.append('')
    out.append('| arm | \\|Δt\\| mean | \\|dx\\| | \\|dy\\| | \\|dz\\| | '
               'dz bias (signed) |')
    out.append('|---|---|---|---|---|---|')
    for name, m, _n in rows2:
        if m is None:
            out.append('| {} | - | - | - | - | - |'.format(name))
            continue
        out.append('| {} | {} | {} | {} | {} | {} |'.format(
            name, fmt(m['dt']), fmt(m['dx']), fmt(m['dy']), fmt(m['dz']),
            fmt(m['dz_signed'])))
    if missing:
        out.append('')
        out.append('(missing reports: {})'.format(', '.join(missing)))
    text = '\n'.join(out)
    print(text)
    if args.output:
        with open(args.output, 'w') as fp:
            fp.write(text + '\n')
        print('\nWritten to {}.'.format(args.output))


if __name__ == '__main__':
    main()
