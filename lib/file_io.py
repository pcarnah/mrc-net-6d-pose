"""Minimal file and image IO helpers.

The project previously used ``mmcv.load``, ``mmcv.dump`` and ``mmcv.imread``.
mmcv 2.x (``mmcv-lite``) dropped those helpers, and only this small subset was
used, so they are reimplemented here to keep the dependency on OpenCV/pickle
directly instead of pulling in mmcv/mmengine.
"""

import json
import pickle

import cv2

_FLAGS = {
    "color": cv2.IMREAD_COLOR,
    "grayscale": cv2.IMREAD_GRAYSCALE,
    "unchanged": cv2.IMREAD_UNCHANGED,
}


def load(path):
    """Load a ``.json`` or ``.pkl``/``.pickle`` file based on its extension."""
    path = str(path)
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError("Unsupported file format for load: {}".format(path))


def dump(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    """Dump ``obj`` to a ``.json`` or ``.pkl``/``.pickle`` file."""
    path = str(path)
    if path.endswith((".pkl", ".pickle")):
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=protocol)
        return
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f)
        return
    raise ValueError("Unsupported file format for dump: {}".format(path))


def imread(path, flag="color", channel_order="bgr"):
    """Read an image with OpenCV.

    ``flag`` is one of ``color``, ``grayscale`` or ``unchanged``. For color
    images ``channel_order`` selects ``bgr`` (OpenCV default) or ``rgb``.
    """
    if flag not in _FLAGS:
        raise ValueError("Unsupported imread flag: {}".format(flag))
    image = cv2.imread(str(path), _FLAGS[flag])
    if image is None:
        raise FileNotFoundError("cv2.imread failed to read: {}".format(path))
    if flag == "color" and channel_order.lower() == "rgb":
        image = image[:, :, ::-1]
    return image
