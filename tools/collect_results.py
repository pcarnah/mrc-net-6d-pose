"""Aggregate per-run eval JSONs into per-step curves and an accuracy-vs-latency
table (plus a dependency-free SVG scatter).

Each eval JSON is produced by ``tools/eval_stereobj.py``.  The manifest is a
JSON list of entries::

    [
      {"name": "stereo_cross_K1", "model": "stereo_cross", "K": 1,
       "eval": "reports/stereo_cross_K1.json"},
      ...
    ]

``model`` groups the per-step (K) curve; ``K`` orders it.  Latency comes from
the eval JSON's ``latency_mean_s`` (measured in the inference job).
"""
import argparse
import json
import os


def load_manifest(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def load_entries(manifest):
    rows = []
    for e in manifest:
        with open(e['eval'], 'r') as fp:
            rep = json.load(fp)
        overall = rep.get('overall', {})
        rows.append({
            'name': e['name'],
            'model': e.get('model', e['name']),
            'K': int(e.get('K', 1)),
            'stereo_mode': rep.get('stereo_mode', ''),
            'n': overall.get('n', 0),
            'add_s_mean': overall.get('add_s_mean', float('nan')),
            'add_s_auc': overall.get('add_s_auc', float('nan')),
            'add_s_acc': overall.get('add_s_acc', float('nan')),
            'proj_mean': overall.get('proj_mean', float('nan')),
            'proj_acc': overall.get('proj_acc', float('nan')),
            'latency_s': rep.get('latency_mean_s', float('nan')),
        })
    return rows


def write_csv(rows, path):
    cols = ['name', 'model', 'K', 'stereo_mode', 'n', 'add_s_mean', 'add_s_auc',
            'add_s_acc', 'proj_mean', 'proj_acc', 'latency_s']
    with open(path, 'w') as fp:
        fp.write(','.join(cols) + '\n')
        for r in rows:
            fp.write(','.join(
                ('{:.6g}'.format(r[c]) if isinstance(r[c], float) else str(r[c]))
                for c in cols) + '\n')


def print_markdown(rows):
    print('| model | K | mode | n | ADD-S (mm) | AUC | ADD-S Acc | '
          'Proj (px) | Proj Acc | Latency (s) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(rows, key=lambda x: (x['model'], x['K'])):
        print('| {} | {} | {} | {} | {:.2f} | {:.3f} | {:.3f} | {:.2f} | '
              '{:.3f} | {:.4f} |'.format(
                  r['model'], r['K'], r['stereo_mode'], r['n'],
                  r['add_s_mean'], r['add_s_auc'], r['add_s_acc'],
                  r['proj_mean'], r['proj_acc'], r['latency_s']))


def write_svg(rows, path, x_key='latency_s', y_key='add_s_acc'):
    """Minimal dependency-free accuracy-vs-latency scatter."""
    pts = [r for r in rows if r[x_key] == r[x_key] and r[y_key] == r[y_key]]
    if not pts:
        return
    w, h, pad = 640, 480, 60
    xs = [r[x_key] for r in pts]
    ys = [r[y_key] for r in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x1 = x1 if x1 > x0 else x0 + 1e-6
    y1 = y1 if y1 > y0 else y0 + 1e-6

    def sx(v):
        return pad + (v - x0) / (x1 - x0) * (w - 2 * pad)

    def sy(v):
        return h - pad - (v - y0) / (y1 - y0) * (h - 2 * pad)

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'
             .format(w, h),
             '<rect width="{}" height="{}" fill="white"/>'.format(w, h),
             '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="black"/>'.format(
                 pad, h - pad, w - pad),
             '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="black"/>'.format(
                 pad, pad, h - pad),
             '<text x="{}" y="{}" font-size="12">{}</text>'.format(
                 w // 2, h - 15, x_key),
             '<text x="10" y="{}" font-size="12" transform="rotate(-90 10 {})">'
             '{}</text>'.format(h // 2, h // 2, y_key)]
    for r in pts:
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="steelblue"/>'
                     .format(sx(r[x_key]), sy(r[y_key])))
        parts.append('<text x="{:.1f}" y="{:.1f}" font-size="9">{}</text>'
                     .format(sx(r[x_key]) + 6, sy(r[y_key]) - 6, r['name']))
    parts.append('</svg>')
    with open(path, 'w') as fp:
        fp.write('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='reports')
    parser.add_argument('--x_key', type=str, default='latency_s')
    parser.add_argument('--y_key', type=str, default='add_s_acc')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_entries(load_manifest(args.manifest))
    print_markdown(rows)
    csv_path = os.path.join(args.outdir, 'summary.csv')
    write_csv(rows, csv_path)
    svg_path = os.path.join(args.outdir, 'accuracy_vs_latency.svg')
    write_svg(rows, svg_path, args.x_key, args.y_key)
    print('\nWrote {} and {}.'.format(csv_path, svg_path))


if __name__ == '__main__':
    main()
