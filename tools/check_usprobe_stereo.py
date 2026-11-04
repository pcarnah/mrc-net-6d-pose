"""Validate the usprobe stereo scene pairing.

Scans every matched ``(2k, 2k+1)`` scene pair and reports:

  * image-id sets that differ between the left and right scene,
  * annotation pairs whose poses do not correspond (rotation differs, or the
    translation difference is not a pure x baseline),
  * missing right scenes / a trailing unpaired scene.

With ``--build`` it also constructs the ``UsprobeDataset`` in stereo mode and
checks the sample contract (the ``STEREO_KEYS`` are present, the right crop
matches the left crop shape, and ``|T_right_ref.t|`` equals the per-sample
baseline).

Exit code is 0 when the pairing is clean enough (no missing scenes and every
matched pair has at least one consistent instance); ``--strict`` also fails on
any image-set or pose mismatch.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg  # noqa: E402
import dataset_schema  # noqa: E402
import usprobe_dataset as ud  # noqa: E402


def report_split(split, strict):
    rep = ud.validate_stereo_pairs('usprobe', split)
    print('=== usprobe [{}] ==='.format(split))
    print('  scene pairs          : {}'.format(rep['pairs']))
    print('  annotations (left)   : {}'.format(rep['instances']))
    print('  consistent pairs     : {}'.format(rep['valid_instances']))
    print('  image-set mismatches : {}'.format(len(rep['image_mismatch'])))
    print('  pose mismatches      : {}'.format(len(rep['pose_mismatch'])))
    if rep['missing_scene']:
        print('  MISSING right scenes : {}'.format(rep['missing_scene']))
    if rep['leftover'] is not None:
        print('  unpaired trailing    : {}'.format(rep['leftover']))
    for left, right, missing, extra in rep['image_mismatch'][:10]:
        print('    {}/{}: missing_right={} extra_right={}'.format(
            left, right, missing, extra))
    if len(rep['image_mismatch']) > 10:
        print('    ... {} more'.format(len(rep['image_mismatch']) - 10))

    failed = bool(rep['missing_scene']) or rep['valid_instances'] == 0
    if strict:
        failed = failed or bool(rep['image_mismatch']) \
            or bool(rep['pose_mismatch'])
    return failed


def build_smoke(split):
    """Build the stereo dataset and check a few samples' contract."""
    cfg.set_runtime_overrides('usprobe', stereo_mode='stereo')
    import dataset_factory
    ds = dataset_factory.create_dataset('usprobe', split=split)
    if getattr(ds, 'stereo_mode', 'mono') != 'stereo':
        print('  BUILD FAIL: stereo_mode={!r}'.format(
            getattr(ds, 'stereo_mode', 'mono')))
        return True
    print('  built stereo dataset : {} instances'.format(len(ds)))
    n = min(8, len(ds))
    for i in range(n):
        s = ds[i]
        for key in dataset_schema.STEREO_KEYS:
            if key not in s:
                print('  BUILD FAIL: sample {} missing {}'.format(i, key))
                return True
        if s['roi_image_right'].shape != s['roi_image'].shape:
            print('  BUILD FAIL: sample {} right/left shape mismatch'.format(i))
            return True
        info = ds.dataset_dicts[i]['inst_infos']
        t_l = np.asarray(info['translation'], dtype=np.float64)
        t_r = np.asarray(info['translation_right'], dtype=np.float64)
        base = float(np.linalg.norm(t_l - t_r))
        T = s['T_right_ref'].numpy()
        if abs(float(np.linalg.norm(T[:3, 3])) - base) > 0.05:
            print('  BUILD FAIL: sample {} |T_right_ref.t|={} != baseline '
                  '{}'.format(i, float(np.linalg.norm(T[:3, 3])), base))
            return True
    print('  sample contract      : OK ({} checked)'.format(n))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='all',
                        choices=['train', 'test', 'all'])
    parser.add_argument('--strict', action='store_true',
                        help='fail on any image-set or pose mismatch')
    parser.add_argument('--build', action='store_true',
                        help='also build the stereo dataset and check samples')
    parser.add_argument('--dataset_root', default=None)
    parser.add_argument('--eval_root', default=None)
    args = parser.parse_args()

    cfg.set_roots(args.dataset_root, args.eval_root)

    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    failed = False
    for split in splits:
        failed |= report_split(split, args.strict)
        if args.build:
            failed |= build_smoke(split)
    print('RESULT: {}'.format('FAIL' if failed else 'PASS'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
