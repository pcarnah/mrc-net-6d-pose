"""Dataset adapter factory.

Selects the adapter for a dataset name based on
``config.DATASET_CONFIG[name]['loader']``: ``'bop'`` (default) for the
BOP-layout datasets, ``'webds'`` for webdataset-backed ones (stereobj-1m),
and ``'usprobe'`` for the usprobe mono/stereo adapter.
"""
import config as cfg


def create_dataset(dataset_name, split='train', rank=0):
    loader = cfg.DATASET_CONFIG[dataset_name].get('loader', 'bop')
    if loader == 'webds':
        from stereobj_dataset import StereobjDataset
        return StereobjDataset(dataset_name, split=split, rank=rank)
    if loader == 'usprobe':
        from usprobe_dataset import UsprobeDataset
        return UsprobeDataset(dataset_name, split=split, rank=rank)
    from bop_dataset import BOP_Dataset
    return BOP_Dataset(dataset_name, split=split)
