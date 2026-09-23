"""Per-object CAD constants consolidated into a single registry.

Before this module, every training sample carried a copy of its object's
vertices, symmetry group and correlation matrix through the DataLoader (and
the collate function NaN-padded them per batch).  The registry builds these
once per dataset, pads them to a global ``V_max``/``S_max`` with explicit
masks, and exposes them as *non-persistent* buffers on an ``nn.Module`` so
they move with ``.to(device)`` and are broadcast by DDP while remaining
absent from ``state_dict()`` (old checkpoints must keep loading strictly).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import trimesh

import config as cfg
import utils

SYMMETRY_EPS = 0.01


class ObjectRegistry(nn.Module):
    def __init__(self, dataset_name, model_folder=None):
        super().__init__()
        self.dataset_name = dataset_name
        dataset_cfg = cfg.DATASET_CONFIG[dataset_name]
        if model_folder is None:
            model_folders = dataset_cfg['model_folders']
            # BOP datasets key their folders by split; 'train_pbr' is the
            # canonical entry. Other layouts (e.g. stereobj webdataset)
            # fall back to their first folder.
            if 'train_pbr' in model_folders:
                model_folder = model_folders['train_pbr']
            else:
                model_folder = next(iter(model_folders.values()))
        self.model_folder = os.path.join(
            cfg.DATASET_ROOT, dataset_name, model_folder)
        self.id2cls = dataset_cfg['id2cls']
        self.num_objects = len(self.id2cls)

        self.obj_ids = sorted(self.id2cls, key=lambda o: self.id2cls[o])
        self.faces = []
        # 'stereobj' models are name-keyed (.obj/.xyz/.diameter per object,
        # in metres) instead of the BOP obj_XXXXXX.ply + models_info.json
        # layout (millimetres).
        self.model_format = dataset_cfg.get('model_format', 'bop')
        self._build()

    def _load_stereobj_model(self, obj_name):
        """Load one stereobj-1m object; returns (mesh, model_info)."""
        mesh = trimesh.load(os.path.join(
            self.model_folder, '{}.obj'.format(obj_name)))
        # The .xyz sidecar holds 2048 uniformly-sampled model points; use
        # them for quat-label quantization (full meshes are 10k+ verts and
        # blow up the (N, bins, V, 3) intermediate).
        xyz_path = os.path.join(self.model_folder, '{}.xyz'.format(obj_name))
        quant_pts = (np.loadtxt(xyz_path, dtype=np.float32)
                     if os.path.exists(xyz_path) else
                     mesh.vertices.astype(np.float32))
        with open(os.path.join(
                self.model_folder,
                '{}.diameter'.format(obj_name)), 'r') as fp:
            diameter = float(fp.read().strip())
        info = {'diameter': diameter,  # metres, matches mesh/pose units
                'quant_points': quant_pts}
        return mesh, info

    def _build(self):
        verts_list = []
        counts = []
        corr_list = []
        quant_corr_list = []
        quat_sym_list = []
        trans_sym_list = []
        diam_list = []
        quant_list = []

        for obj_id in self.obj_ids:
            if self.model_format == 'stereobj':
                mesh, info = self._load_stereobj_model(obj_id)
                vertices = torch.from_numpy(
                    mesh.vertices.copy()).to(torch.float32)
                faces = torch.from_numpy(mesh.faces.copy()).to(torch.int64)
                diam_m = info['diameter']
                diam_list.append(np.array(diam_m, dtype=np.float32))
                quant_list.append(torch.from_numpy(
                    info['quant_points']).to(torch.float32))
            else:
                with open(os.path.join(
                        self.model_folder, 'models_info.json'), 'r') as fp:
                    model_info = json.load(fp)
                info = model_info[str(obj_id)]
                model_path = os.path.join(
                    self.model_folder, 'obj_{:06d}.ply'.format(int(obj_id)))
                mesh = trimesh.load(model_path)
                vertices = torch.from_numpy(
                    mesh.vertices.copy()).to(torch.float32)
                faces = torch.from_numpy(mesh.faces.copy()).to(torch.int64)
                diam_list.append(np.array(
                    info['diameter'], dtype=np.float32))
                # BOP quantization runs on the full mesh vertices.
                quant_list.append(vertices)

            self.faces.append(faces)

            num_v = len(vertices)
            counts.append(num_v)
            verts_list.append(vertices)
            corr_list.append(vertices.T @ vertices / num_v)

            rotations_sym, translations_sym = \
                utils.get_symmetry_transformations(info, SYMMETRY_EPS)
            quat_sym_list.append(utils.rotation_to_quaternion(
                torch.from_numpy(rotations_sym).to(torch.float32)))
            trans_sym_list.append(torch.from_numpy(
                translations_sym).to(torch.float32))

        num_objects = len(verts_list)
        v_max = max(counts)
        s_max = max(len(q) for q in quat_sym_list)

        vertices = torch.zeros(num_objects, v_max, 3, dtype=torch.float32)
        vertices_mask = torch.zeros(num_objects, v_max, dtype=torch.bool)
        quat_sym = torch.zeros(num_objects, s_max, 4, dtype=torch.float32)
        trans_sym = torch.zeros(num_objects, s_max, 3, dtype=torch.float32)
        sym_mask = torch.zeros(num_objects, s_max, dtype=torch.bool)
        q_max = max(len(q) for q in quant_list) if quant_list else 0
        quant_points = torch.zeros(num_objects, q_max, 3,
                                   dtype=torch.float32)
        quant_mask = torch.zeros(num_objects, q_max, dtype=torch.bool)

        for cls in range(num_objects):
            n_v = counts[cls]
            vertices[cls, :n_v] = verts_list[cls]
            vertices_mask[cls, :n_v] = True
            n_s = len(quat_sym_list[cls])
            quat_sym[cls, :n_s] = quat_sym_list[cls]
            trans_sym[cls, :n_s] = trans_sym_list[cls]
            sym_mask[cls, :n_s] = True
            if quant_list:
                n_q = len(quant_list[cls])
                quant_points[cls, :n_q] = quant_list[cls]
                quant_mask[cls, :n_q] = True
                quant_corr_list.append(
                    quant_list[cls].T @ quant_list[cls] / n_q)

        self.register_buffer('vertices', vertices, persistent=False)
        self.register_buffer('vertices_mask', vertices_mask, persistent=False)
        self.register_buffer('num_vertices', torch.tensor(
            counts, dtype=torch.int64), persistent=False)
        self.register_buffer('vertices_correlation', torch.stack(
            corr_list, dim=0), persistent=False)
        self.register_buffer('quaternion_symmetries', quat_sym,
                             persistent=False)
        self.register_buffer('translation_symmetries', trans_sym,
                             persistent=False)
        self.register_buffer('symmetries_mask', sym_mask, persistent=False)
        self.register_buffer('diameter', torch.from_numpy(np.stack(
            diam_list, axis=0)), persistent=False)
        # Quantization points (stereobj .xyz samples; == vertices for BOP).
        self.register_buffer('quant_points', quant_points, persistent=False)
        self.register_buffer('quant_points_mask', quant_mask,
                             persistent=False)
        self.register_buffer('quant_correlation', torch.stack(
            quant_corr_list, dim=0), persistent=False)

    def get_vertices(self, cls):
        """Unpadded vertices of the object with class index ``cls``."""
        return self.vertices[cls][:self.num_vertices[cls]]

    def get_faces(self, cls):
        return self.faces[cls]

    def gather(self, obj_cls):
        """Return the CAD tensors consumed by the losses for a batch."""
        obj_cls = obj_cls.to(self.vertices.device)
        return {
            'vertices': self.vertices[obj_cls],
            'vertices_mask': self.vertices_mask[obj_cls],
            'vertices_correlation': self.vertices_correlation[obj_cls],
            'quaternion_symmetries': self.quaternion_symmetries[obj_cls],
            'translation_symmetries': self.translation_symmetries[obj_cls],
            'symmetries_mask': self.symmetries_mask[obj_cls],
            'diameter': self.diameter[obj_cls],
        }