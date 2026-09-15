"""Multi-view-capable renderer protocol.

The renderer is the only component that differs between mono (current),
stereo (two views) and future video (T views) inference, so it is isolated
behind ``RendererProtocol``.  ``Pytorch3DRenderer`` implements it on top of
pytorch3d, consuming the mesh store from ``ObjectRegistry`` (no PLY loading
happens here).

Numerics are kept bit-identical to the previous
``utils.SyntheticRenderer`` for a single identity view: same lights,
materials, shader parameters, negated focal lengths and ``cam_pose = R``.
"""
from dataclasses import dataclass
from typing import List, Optional, Protocol

import torch
from pytorch3d import renderer as p3d_renderer
from pytorch3d import structures as p3d_struct

from utils import image_to_bbox_map


@dataclass
class CameraView:
    """One camera of a multi-view rig.

    K: intrinsics.  Either a single ``(3, 3)`` matrix shared by the whole
       batch, or ``(N, 3, 3)`` for per-sample intrinsics (the usual case for
       object-centric crops).
    T_view_ref: ``(4, 4)`` transform from the reference (left/mono) camera
       frame into this view's frame.  ``None`` means identity (mono).
    """
    K: torch.Tensor
    T_view_ref: Optional[torch.Tensor] = None

    def matrix(self, device):
        if self.T_view_ref is None:
            return torch.eye(4, device=device)
        return self.T_view_ref.to(device)


class RendererProtocol(Protocol):
    def render(self, obj_cls, R_obj, t_obj,
               views: List[CameraView]) -> dict:
        """Render ``obj_cls`` at poses ``(R_obj, t_obj)`` from every view.

        Returns ``{'rgb': (N,V,3,H,W), 'mask': (N,V,1,H,W),
        'bbox_map': (N,V,1,H,W)}`` under ``torch.no_grad()``.
        """
        ...


class Pytorch3DRenderer:
    def __init__(self, image_width, image_height, registry,
                 surf_color=(0.6, 0.6, 0.6)):
        self.width = image_width
        self.height = image_height
        self.registry = registry
        self.surf_color = list(surf_color)
        self._blend_params = p3d_renderer.BlendParams(
            background_color=[0., 0., 0.])

    def _intrinsics_for(self, view, index):
        K = view.K
        if K.dim() == 3:
            K = K[index]
        return K.to(torch.float32)

    @torch.no_grad()
    def render(self, obj_cls, R_obj, t_obj, views):
        device = R_obj.device
        n = len(obj_cls)
        n_views = len(views)
        R_obj = R_obj.to(torch.float32)
        t_obj = t_obj.to(torch.float32)

        view_R = []
        view_t = []
        for view in views:
            T = view.matrix(device).to(torch.float32)
            view_R.append(T[:3, :3])
            view_t.append(T[:3, 3])

        # Mesh batch replicated x n_views (sample-major, then view).
        verts_list, faces_list, colors_list = [], [], []
        for i in obj_cls:
            verts = self.registry.get_vertices(i)
            faces = self.registry.get_faces(i).to(device)
            colors = torch.tensor(
                [self.surf_color], device=device).expand(len(verts), 3)
            for _ in range(n_views):
                verts_list.append(verts)
                faces_list.append(faces)
                colors_list.append(colors)
        textures = p3d_renderer.TexturesVertex(verts_features=colors_list)
        meshes = p3d_struct.Meshes(
            verts=verts_list, faces=faces_list,
            textures=textures).to(device)

        batch = n * n_views
        focal_length, principal_point = [], []
        cam_pose, cam_trans, light_direction = [], [], []
        for ci in range(n):
            for vi in range(n_views):
                R_v = view_R[vi] @ R_obj[ci]
                t_v = (view_R[vi] @ t_obj[ci].unsqueeze(-1)).squeeze(-1) \
                    + view_t[vi]
                K = self._intrinsics_for(views[vi], ci)
                focal_length.append(torch.stack([-K[0, 0], -K[1, 1]]))
                principal_point.append(torch.stack([K[0, 2], K[1, 2]]))
                cam_pose.append(torch.transpose(R_v, 0, 1))
                cam_trans.append(t_v)
                light_direction.append(torch.squeeze(
                    -t_v.unsqueeze(-2) @ R_v, dim=-2))

        focal_length = torch.stack(focal_length, dim=0).to(torch.float32)
        principal_point = torch.stack(principal_point, dim=0).to(torch.float32)
        cam_pose = torch.stack(cam_pose, dim=0)
        cam_trans = torch.stack(cam_trans, dim=0)
        light_direction = torch.stack(light_direction, dim=0)

        image_size = (self.height, self.width)
        materials = p3d_renderer.Materials(
            device=device,
            diffuse_color=[self.surf_color for _ in range(batch)],
            specular_color=[self.surf_color for _ in range(batch)],
            shininess=10.0)
        cameras = p3d_renderer.cameras.PerspectiveCameras(
            focal_length, principal_point, cam_pose, cam_trans,
            device=device, in_ndc=False, image_size=(image_size,))
        lights = p3d_renderer.DirectionalLights(
            ambient_color=[[0.8, 0.8, 0.8]], diffuse_color=[[0.5, 0.5, 0.5]],
            specular_color=[[0.1, 0.1, 0.1]],
            direction=light_direction, device=device)

        raster_settings = p3d_renderer.RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = p3d_renderer.MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings)
        shader = p3d_renderer.HardPhongShader(
            device=device, cameras=cameras, lights=lights, materials=materials,
            blend_params=self._blend_params)
        renderer = p3d_renderer.MeshRendererWithFragments(rasterizer, shader)

        images, fragments = renderer(meshes)
        rgb = images[..., :3].permute(0, 3, 1, 2)
        depth = fragments.zbuf.permute(0, 3, 1, 2)
        mask = (depth[:, :1] > 0).float()
        bbox_map = image_to_bbox_map(depth).unsqueeze(1)

        return {
            'rgb': rgb.reshape(n, n_views, 3, self.height, self.width),
            'mask': mask.reshape(n, n_views, 1, self.height, self.width),
            'bbox_map': bbox_map.reshape(
                n, n_views, 1, self.height, self.width),
        }