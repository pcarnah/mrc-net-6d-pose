"""Paired frame-level bootstrap for arm comparisons on a matched subset.

All arms scored the identical stratified instance set, so per-instance
records join 1:1 across arms by (scene_id, im_id, obj_id) and a bootstrap
resample of FRAMES yields paired differences: the same frames are drawn for
both arms of a pair, and the statistic is the difference of per-arm
aggregates. This is far more sensitive than overlapping independent CIs when
per-arm spreads are dominated by frame-to-frame variation.

Aggregation follows the summary-table protocol: per-object mean, then
unweighted mean over objects (macro), excluding one object (default
``centrifuge_tube``).

Pair syntax: ``A[:B]`` with an optional per-side suffix override
``run_tag@suffix`` (default ``--suffix``). The diff is A - B.
"""
import argparse
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
import ablation_table as at  # noqa: E402


def load_arm_records(csv_path, gt, cls_names, registry, syms, max_points,
                     exclude):
    """Per-instance records for one arm, sorted by (scene, im, obj) key.

    Returns (keys, frames, add_s, add_ok, dz) arrays plus the per-object
    boolean masks over record positions (excluded object dropped).
    """
    per_key = {}
    for res in inout.load_bop_results(csv_path):
        key = (int(res['scene_id']), int(res['im_id']), int(res['obj_id']))
        g = gt.get(key)
        if g is None:
            continue
        cls = g['cls']
        name = cls_names[cls]
        if name == exclude:
            continue
        pts = es.subsample(
            registry.get_vertices(cls).cpu().numpy().astype(np.float64),
            max_points)
        R_est = np.asarray(res['R'], dtype=np.float64)
        t_est = np.asarray(res['t'], dtype=np.float64).reshape(3)
        sym = syms.get(name, [np.eye(4)])
        if not any(np.allclose(s, np.eye(4)) for s in sym):
            sym = [np.eye(4)] + list(sym)
        add = es.add_s(R_est, t_est, g['R'], g['t'], pts, sym)
        per_key[key] = (key[0] * 1000000 + key[1], add,
                        add < es.ADD_THRESH_FRAC * float(registry.diameter[cls]),
                        abs(t_est[2] - g['t'][2]))
    keys = sorted(per_key)
    frames = np.array([per_key[k][0] for k in keys], dtype=np.int64)
    add_s = np.array([per_key[k][1] for k in keys], dtype=np.float64)
    add_ok = np.array([per_key[k][2] for k in keys], dtype=np.bool_)
    dz = np.array([per_key[k][3] for k in keys], dtype=np.float64)
    objs = np.array([cls_names[k[2] - 1] for k in keys])
    masks = {o: (objs == o) for o in sorted(set(objs))}
    return keys, frames, add_s, add_ok, dz, masks


def macro_over_objects(values, masks, ok_mask=None):
    """Unweighted mean of per-object means (empty objects skipped)."""
    out, used = [], 0
    for o, m in masks.items():
        v = values[m]
        if ok_mask is not None:
            v = v[ok_mask[m]]
        if len(v):
            out.append(v.mean())
            used += 1
    return float(np.mean(out)) if out else float('nan'), used


def frame_gather_tables(frames):
    """Sort records by frame; return (order, starts, counts, n_frames)."""
    order = np.argsort(frames, kind='stable')
    f_sorted = frames[order]
    uniq, starts = np.unique(f_sorted, return_index=True)
    counts = np.diff(np.append(starts, len(f_sorted)))
    return order, starts, counts, len(uniq)


def paired_bootstrap(arm_data, pair, metrics, n_bootstrap, seed):
    """Bootstrap the macro-metric difference A - B over resampled frames."""
    rng = np.random.default_rng(seed)
    a_tag, b_tag = pair
    a, b = arm_data[a_tag], arm_data[b_tag]
    if a['keys'] != b['keys']:
        raise SystemExit('pair {}: key sets differ ({} vs {} instances); '
                         'the arms were not run on the identical '
                         'subset'.format(pair, len(a['keys']), len(b['keys'])))
    order, starts, counts, n_frames = a['gather']

    samples = {m: [] for m in metrics}
    for _ in range(n_bootstrap):
        pick = rng.integers(0, n_frames, size=n_frames)
        counts_p = counts[pick]
        starts_p = starts[pick]
        total = int(counts_p.sum())
        cum = np.cumsum(counts_p)
        gstart = cum - counts_p
        within = np.arange(total) - np.repeat(gstart, counts_p)
        idx = np.repeat(starts_p, counts_p) + within
        pos = order[idx]  # positions into the canonical record order
        for m in metrics:
            diffs = []
            for side in (a, b):
                per_obj = []
                for o, m_obj in side['masks'].items():
                    sel = pos[m_obj[pos]]
                    if len(sel):
                        per_obj.append(side[m][sel].mean())
                diffs.append(float(np.mean(per_obj)))
            samples[m].append(diffs[0] - diffs[1])
    out = {}
    for m in metrics:
        s = np.array(samples[m])
        obs_a, _ = macro_over_objects(a[m], a['masks'])
        obs_b, _ = macro_over_objects(b[m], b['masks'])
        out[m] = {
            'a': obs_a, 'b': obs_b, 'diff': obs_a - obs_b,
            'ci': [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))],
            'p_gt0': float((s > 0).mean()),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports_dir', type=str,
                        default=os.path.join(ROOT, 'reports',
                                             'ablation_10k_multi'))
    parser.add_argument('--dataset', type=str, default='stereobj-1m')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--suffix', type=str, default='n10000_strat_multi')
    parser.add_argument('--pairs', nargs='+', required=True,
                        help='A:B (diff = A - B); sides may override the '
                             'suffix with run_tag@suffix')
    parser.add_argument('--metrics', nargs='+',
                        default=['add_s_mean', 'add_s_acc', 'dz_mean'])
    parser.add_argument('--exclude', type=str, default='centrifuge_tube')
    parser.add_argument('--max_points', type=int, default=1000)
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    cfg.set_roots(args.dataset_root)
    cfg.set_runtime_overrides(args.dataset, stereo_mode='stereo')
    ds = dataset_factory.create_dataset(args.dataset, split=args.split)
    gt = es.build_gt(ds, stereo=True)
    syms = es.load_symmetries(args.dataset, None)

    # Resolve every (run_tag, suffix) combo once.
    combos = {}
    for pair in args.pairs:
        for side in pair.split(':'):
            tag, _, sfx = side.partition('@')
            combos.setdefault((tag, sfx or args.suffix), None)
    arm_data = {}
    for (tag, sfx) in combos:
        _, csv_path = at._resolve_tags(
            args.reports_dir, args.dataset, args.split, tag, 'epoch_000', sfx)
        if csv_path is None:
            raise SystemExit('no CSV for {} (suffix {}) in {}'.format(
                tag, sfx, args.reports_dir))
        keys, frames, add_s, add_ok, dz, masks = load_arm_records(
            csv_path, gt, ds.cls_names, ds.registry, syms, args.max_points,
            args.exclude)
        gather = frame_gather_tables(frames)
        arm_data[(tag, sfx)] = {
            'keys': keys, 'masks': masks, 'gather': gather,
            'add_s_mean': add_s, 'add_s_acc': add_ok.astype(np.float64),
            'dz_mean': dz,
        }
        print('{} ({}): {} records, {} objects, {} frames'.format(
            tag, sfx, len(keys), len(masks), gather[3]))

    rows = []
    for pair in args.pairs:
        a_raw, b_raw = pair.split(':')
        a_tag, _, a_sfx = a_raw.partition('@')
        b_tag, _, b_sfx = b_raw.partition('@')
        a_key = (a_tag, a_sfx or args.suffix)
        b_key = (b_tag, b_sfx or args.suffix)
        res = paired_bootstrap(
            {a_key: arm_data[a_key], b_key: arm_data[b_key]},
            (a_key, b_key), args.metrics, args.n_bootstrap, args.seed)
        for m in args.metrics:
            r = res[m]
            sig = 'yes' if (r['ci'][0] > 0 or r['ci'][1] < 0) else 'no'
            rows.append(('{} − {}'.format(a_tag, b_tag), m, r, sig))

    lines = ['Paired frame-level bootstrap ({} resamples, seed {}; macro '
             'over objects, excluding {}; diff = A − B)'.format(
                 args.n_bootstrap, args.seed, args.exclude), '']
    lines.append('| pair (A − B) | metric | A | B | Δ obs | 95% CI | CI excludes 0 |')
    lines.append('|---|---|---|---|---|---|---|')
    for pair_name, m, r, sig in rows:
        nd = 3 if m == 'add_s_acc' else 2
        fmt = lambda v, sign=False: '{:+.{}f}'.format(v, nd) if sign \
            else '{:.{}f}'.format(v, nd)
        ci = '[{}, {}]'.format(fmt(r['ci'][0], True), fmt(r['ci'][1], True))
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            pair_name, m, fmt(r['a']), fmt(r['b']), fmt(r['diff'], True),
            ci, sig))
    text = '\n'.join(lines)
    print(text)
    if args.output:
        with open(args.output, 'w') as fp:
            fp.write(text + '\n')
        print('\nWritten to {}.'.format(args.output))


if __name__ == '__main__':
    main()
