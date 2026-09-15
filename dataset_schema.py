"""Minimal data -> model interface for Phase 0.

`PoseSample` is the documented dict contract every dataset adapter must
return from ``__getitem__``.  `PoseDataset` is the adapter ABC.

Layout (batch dimension added by the collate function):

    inputs
        roi_image  float32 (3, H, W)
        bbox_map   float32 (1, H, W)
        roi_camK   float32 (3, 3)
        fov        float32 (3,)
        obj_cls    int64   ()
    targets
        roi_obj_R  float32 (3, 3)   Rz-composed ground-truth rotation
        roi_obj_t  float32 (3,)     ground-truth translation
        roi_mask   float32 (M, M)
        quat_bin   float32 (N_POSE_BIN,)
    meta (ints, never moved to the GPU as targets)
        obj_id, scene_id, image_id
"""
from abc import ABC, abstractmethod
from typing import TypedDict

import torch

INPUT_KEYS = ['roi_image', 'bbox_map', 'roi_camK', 'fov', 'obj_cls']
TARGET_KEYS = ['roi_obj_R', 'roi_obj_t', 'roi_mask', 'quat_bin']
META_KEYS = ['obj_id', 'scene_id', 'image_id']


class PoseSample(TypedDict):
    # inputs
    roi_image: torch.Tensor
    bbox_map: torch.Tensor
    roi_camK: torch.Tensor
    fov: torch.Tensor
    obj_cls: torch.Tensor
    # targets
    roi_obj_R: torch.Tensor
    roi_obj_t: torch.Tensor
    roi_mask: torch.Tensor
    quat_bin: torch.Tensor
    # meta
    obj_id: int
    scene_id: int
    image_id: int


class PoseDataset(ABC):
    """Common interface for pose datasets feeding ``MRCNet``."""

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx) -> PoseSample:
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> dict:
        """Return {'depth_min', 'depth_max', 'num_objects'}."""
        raise NotImplementedError