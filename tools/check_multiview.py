"""Multi-view renderer check.

Renders a fixed pose from two views (identity + a 5 cm x-shift baseline) in
one pytorch3d call and asserts that view 2 matches a direct mono render of
the *composed* pose ``(R_RL @ R, R_RL @ t + t_RL)``.
"""
import os
import sys
import math

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg  # noqa: E402
import rendering  # noqa: E402
from object_registry import ObjectRegistry  # noqa: E402

RGB_RTOL = 1e-3
MASK_IOU_MIN = 0.999


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.RANDOM_SEED)

    registry = ObjectRegistry('usprobe').to(device)
    renderer = rendering.Pytorch3DRenderer(
        cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE, registry)

    batch = 2
    obj_cls = torch.tensor([0, 1], dtype=torch.int64, device=device)

    # Object coordinates and translations are in millimetres.
    R_obj = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    base_t = torch.tensor([
        [0.0, 0.0, 500.0],
        [0.0, 0.0, 550.0],
    ], dtype=torch.float32, device=device)
    means = torch.stack(
        [registry.get_vertices(int(c)).mean(dim=0) for c in obj_cls])
    t_obj = base_t - means

    K = torch.tensor([
        [407.35, 0., 128.0],
        [0., 407.35, 128.0],
        [0., 0., 1.]], dtype=torch.float32, device=device)
    K_left = K.unsqueeze(0).repeat(batch, 1, 1)
    K_right = K_left.clone()
    K_right[:, 0, 0] += 8.0

    # 5 cm x-shift plus a small 1 degree y-rotation of the right camera.
    angle = 1.0 * math.pi / 180.0
    R_RL = torch.tensor([
        [math.cos(angle), 0., math.sin(angle)],
        [0., 1., 0.],
        [-math.sin(angle), 0., math.cos(angle)]],
        dtype=torch.float32, device=device)
    t_RL = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float32, device=device)
    T_RL = torch.eye(4, device=device)
    T_RL[:3, :3] = R_RL
    T_RL[:3, 3] = t_RL

    views = [
        rendering.CameraView(K=K_left),
        rendering.CameraView(K=K_right, T_view_ref=T_RL),
    ]
    out = renderer.render(obj_cls, R_obj, t_obj, views)
    assert out['rgb'].shape == (batch, 2, 3, cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE)
    assert out['mask'].shape == (batch, 2, 1, cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE)
    assert out['bbox_map'].shape == (batch, 2, 1, cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE)

    # Direct mono render of the composed right-camera pose.
    R_composed = R_RL @ R_obj
    t_composed = (R_RL @ t_obj.unsqueeze(-1)).squeeze(-1) + t_RL
    mono = renderer.render(
        obj_cls, R_composed, t_composed,
        [rendering.CameraView(K=K_right)])

    rgb_multi = out['rgb'][:, 1]
    rgb_mono = mono['rgb'][:, 0]
    rgb_ok = torch.allclose(rgb_multi, rgb_mono, rtol=RGB_RTOL, atol=1e-4)
    rgb_diff = float((rgb_multi - rgb_mono).abs().max())

    m_multi = out['mask'][:, 1, 0] > 0.5
    m_mono = mono['mask'][:, 0, 0] > 0.5
    assert m_multi.sum() > 0, 'render produced an empty mask; pose not visible'
    inter = (m_multi & m_mono).sum(dim=(-1, -2)).float()
    union = (m_multi | m_mono).sum(dim=(-1, -2)).float()
    iou = float((inter / union.clamp(min=1)).min())

    print('view-2 vs composed mono: rgb max_abs_diff={:.3e} allclose={} '
          'mask IoU={:.5f}'.format(rgb_diff, rgb_ok, iou))

    if not rgb_ok or iou <= MASK_IOU_MIN:
        print('MULTIVIEW CHECK FAILED')
        sys.exit(1)
    print('Multi-view check passed.')


if __name__ == '__main__':
    main()