import torchvision

import utils
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import models as model_zoo
import config as cfg
from functools import partial
from correlation import SpatialCorrelationSampler
from object_registry import ObjectRegistry
import rendering


class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, norm_groups=32):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class CondConv(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size,
                 stride=1, padding=None, norm=False,
                 n_class=1, activation='swish'):
        super(CondConv, self).__init__()
        self.weights = nn.Parameter(torch.empty(
            n_class, n_filters, n_channels, kernel_size, kernel_size))
        if norm:
            self.norm = nn.GroupNorm(num_groups=32, num_channels=n_filters)
            self.bias = None
        else:
            self.norm = None
            self.bias = nn.Parameter(torch.empty(
                n_class, n_filters, 1, 1))
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding if padding is not None else kernel_size // 2

        for i, weight in enumerate(self.weights):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            if fan_in != 0 and self.bias is not None:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input, decoder_idx):
        N, Cin, Hi, Wi = input.shape
        _, Cout, Cin, Hk, Wk = self.weights.shape
        x = torch.reshape(input, (1, N*Cin, Hi, Wi))
        weight = torch.reshape(
            self.weights[decoder_idx], (N*Cout, Cin, Hk, Wk))
        x = F.conv2d(x, weight, bias=None, stride=self.stride,
                     padding=self.padding, groups=N)
        x = torch.reshape(x, (N, Cout, *x.shape[2:]))
        if self.bias is not None:
            bias = self.bias[decoder_idx]
            x = x + bias
        if self.norm is not None:
            x = self.norm(x)
        if self.activation == 'relu':
            x = F.relu(x, inplace=True)
        elif self.activation == 'swish':
            x = F.silu(x, inplace=True)
        return x


class ResNet34_AsymUNet(nn.Module):
    def __init__(self, out_feat_dim=64, rgb_input_dim=3,
                 n_decoders=1, track_running_stats=False):
        super().__init__()
        self.out_feat_dim = out_feat_dim
        norm_layer = partial(
            nn.BatchNorm2d, track_running_stats=track_running_stats)
        model = model_zoo.resnet34(norm_layer=norm_layer)
        # ResNet34_Weights_IMAGENET1K_V1 = torch.hub.load_state_dict_from_url(
        #     'https://download.pytorch.org/models/resnet34-b627a593.pth')
        # model.load_state_dict(ResNet34_Weights_IMAGENET1K_V1, strict=False)
        ResNet34_Weights_IMAGENET1K_V1 = model_zoo.resnet34(weights=model_zoo.ResNet34_Weights.IMAGENET1K_V1).state_dict()
        model.load_state_dict(ResNet34_Weights_IMAGENET1K_V1, strict=False)
        # Remove last pooling and fc layer
        self.base_model = torch.nn.Sequential(*(list(model.children())[:-2]))

        if rgb_input_dim != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(rgb_input_dim, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64, track_running_stats=False),
                nn.SiLU(inplace=True))  # CxHxW  -> 64xH/2xW/2
        else:
            self.layer0 = nn.Sequential(
                *list(self.base_model.children())[0:3])  # 3xHxW  -> 64xH/2xW/2

        # 64xH/2xW/2   -> 64xH/2xW/2
        self.layer1 = nn.Sequential(*list(self.base_model.children())[4:5])
        # 64xH/2xW/2   -> 128xH/4xW/4
        self.layer2 = nn.Sequential(*list(self.base_model.children())[5:6])
        # 128xH/4xW/4  -> 256xH/8xW/8
        self.layer3 = nn.Sequential(*list(self.base_model.children())[6:7])
        # 256xH/8xW/8  -> 512xH/16xW/16
        self.layer4 = nn.Sequential(*list(self.base_model.children())[7:8])
        self.pool = nn.Sequential(
            AtrousSpatialPyramidPooling(
                n_features=512, dilation_rates=[2, 3, 4, 6, 8, 12]),
            nn.Conv2d(512 * 8, 512, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.SiLU(inplace=True))

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        # decoder
        self.decoder = nn.ModuleDict(dict(
            layer2_1x1=CondConv(
                128, 128, kernel_size=1, stride=1, n_class=n_decoders),
            layer3_1x1=CondConv(
                256, 256, kernel_size=1, stride=1, n_class=n_decoders),
            layer4_1x1=CondConv(
                512, 512, kernel_size=1, stride=1, n_class=n_decoders),
            conv_up3=CondConv(256 + 512, 512, kernel_size=3,
                              stride=1, padding=1, n_class=n_decoders),
            conv_up2=CondConv(128 + 512, 256, kernel_size=3,
                              stride=1, padding=1, n_class=n_decoders),
            conv_last=CondConv(256, self.out_feat_dim + 1,
                               kernel_size=1, stride=1,
                               activation=None, n_class=n_decoders)))

    def forward(self, input, decoder_idx):
        # encoder
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        # decoders
        batch_size, _, in_height, in_width = input.shape

        stem_x = self.pool(layer4)
        # Bx512x16x16 => Bx512x16x16
        stem_x = self.decoder['layer4_1x1'](stem_x, decoder_idx)
        # Bx512x16x16 => Bx512x32x32
        stem_x = self.upsample(stem_x)
        layer_slice = layer3        # Bx256x32x32
        # Bx256x32x32 =>  # Bx256x32x32
        layer_projection = self.decoder['layer3_1x1'](
            layer_slice, decoder_idx)
        stem_x = torch.cat([stem_x, layer_projection],
                           dim=1)  # Bx(512+256)x32x32
        # Bx(512+256)x32x32 => Bx512x32x32
        stem_x = self.decoder['conv_up3'](stem_x, decoder_idx)
        # Bx512x32x32 => Bx512x64x64
        stem_x = self.upsample(stem_x)

        # Bx128x64x64
        layer_slice = layer2
        # Bx128x64x64 => Bx128x64x64
        layer_projection = self.decoder['layer2_1x1'](
            layer_slice, decoder_idx)
        stem_x = torch.cat([stem_x, layer_projection],
                           dim=1)  # Bx(512+128)x64x64
        # Bx(512+128)x64x64 => Bx256x64x64
        stem_x = self.decoder['conv_up2'](
            stem_x, decoder_idx)
        # Bx256x64x64 => BxCx64x64
        x_out = self.decoder['conv_last'](stem_x, decoder_idx)

        rgb_emb = x_out[:, :-1]         # BxCx64x64
        visib_msk = x_out[:, -1:]       # Bx1x64x64
        return rgb_emb, visib_msk


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, n_features=128, dilation_rates=[2, 3, 4, 5, 6, 7]):
        super(AtrousSpatialPyramidPooling, self).__init__()

        self.convs = nn.ModuleList([
            ConvGnReLU(n_features, n_features, kernel_size=3,
                       stride=1, dilation=rate, padding=rate)
            for rate in dilation_rates])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        x = inputs
        x_pool = self.global_pool(x)
        # BxCx1x1 => BxCxH/16xW/16
        x_pool = x_pool.repeat(1, 1, *x.shape[-2:])
        x_out = torch.cat(
            [x, x_pool] + [conv(x) for conv in self.convs], dim=1)
        return x_out


class Correlation(nn.Module):
    def __init__(self, patch_size, input_dim=128, feature_dim=128):
        super(Correlation, self).__init__()
        self.proj_in = nn.Conv2d(input_dim, feature_dim, kernel_size=1,
                                 stride=1, padding=0)
        self.corr = SpatialCorrelationSampler(patch_size=patch_size)
        self.feature_dim = feature_dim

    def forward(self, input1, input2):
        x1 = self.proj_in(input1)
        x2 = self.proj_in(input2)
        output = self.corr(x1, x2)
        output = output.view(
            len(input1), -1, *input1.shape[-2:])
        output = output / math.sqrt(self.feature_dim)

        return output


class PoseDecoder(nn.Module):
    def __init__(self):
        super(PoseDecoder, self).__init__()
        self.conv_down_x8 = ConvGnReLU(
            64 + 1, 128, kernel_size=3, stride=2, padding=1)
        self.conv_down_x16 = ConvGnReLU(
            128, 128, kernel_size=3, stride=2, padding=1)

        self.clf_pool = AtrousSpatialPyramidPooling(
            n_features=128, dilation_rates=[2, 3, 4, 6, 8, 12])

        self.corr_x4 = Correlation(
            patch_size=11, input_dim=64 + 1, feature_dim=64)
        self.reg_conv_x8 = ConvGnReLU(
            64 + 1 + 121, 128, kernel_size=3, stride=2, padding=1)
        self.corr_x8 = Correlation(
            patch_size=11, input_dim=128, feature_dim=128)
        self.reg_conv_x16 = ConvGnReLU(
            128 + 121 + 128, 256, kernel_size=3, stride=2, padding=1)
        self.corr_x16 = Correlation(
            patch_size=11, input_dim=128, feature_dim=128)
        self.reg_conv_add = ConvGnReLU(
            128 + 121 + 256, 256, kernel_size=3, stride=1, padding=1)

        self.downsample = nn.AvgPool2d(2)
        self.reg_pool = AtrousSpatialPyramidPooling(
            n_features=256, dilation_rates=[2, 3, 4, 6, 8, 12])

    def forward(self, x_real, x_synt=None):
        if x_synt is None:
            x_real_x4 = x_real
            # BxCxH/4xW/4 => Bx128xH/8xW/8
            x_real_x8 = self.conv_down_x8(x_real_x4)
            # Bx128xH/8xW/8 => Bx128xH/16xW/16
            x_real_x16 = self.conv_down_x16(x_real_x8)
            # Bx128xH/16xW/16 => Bx128*8xH/16xW/16
            x_clf_pool = self.clf_pool(x_real_x16)
            predictions = {
                'real_x4': x_real_x4,
                'real_x8': x_real_x8,
                'real_x16': x_real_x16,
                'clf_pool': x_clf_pool}
            return predictions
        else:
            x_real_x4 = x_real['real_x4']
            x_real_x8 = x_real['real_x8']
            x_real_x16 = x_real['real_x16']
            x_synt_x4 = x_synt

            x_corr_x4 = self.corr_x4(x_synt_x4, x_real_x4)
            # Bx(C+121)xH/4xW/4 => Bx128xH/8xW/8
            x_reg_x8 = self.reg_conv_x8(
                torch.concat([x_real_x4, x_corr_x4], dim=1))

            # BxCxH/4xW/4 => Bx128xH/8xW/8
            x_synt_x8 = self.conv_down_x8(x_synt_x4)
            x_corr_x8 = self.corr_x8(x_synt_x8, x_real_x8)

            # Bx(121+256)xH/8xW/8 => Bx256xH/16xW/16
            x_reg_x16 = self.reg_conv_x16(
                torch.concat([x_real_x8, x_corr_x8, x_reg_x8], dim=1))
            # Bx128xH/8xW/8 => Bx128xH/16xW/16
            x_synt_x16 = self.conv_down_x16(x_synt_x8)
            x_corr_x16 = self.corr_x16(x_synt_x16, x_real_x16)
            # Bx(121+384)xH/16xW/16 => Bx256xH/16xW/16
            x_reg_add = self.reg_conv_add(
                torch.concat([x_real_x16, x_corr_x16, x_reg_x16], dim=1))
            # Bx256xH/16xW/16 => Bx256*8xH/16xW/16
            x_reg_pool = self.reg_pool(x_reg_add)
            predictions = {
                'synt_x4': x_synt_x4,
                'synt_x8': x_synt_x8,
                'reg_x8': x_reg_x8,
                'synt_x16': x_synt_x16,
                'reg_x16': x_reg_x16,
                'reg_pool': x_reg_pool}
            return predictions


class TaskHead(nn.Module):
    def __init__(self, n_features, n_class, hidden_size=256):
        super(TaskHead, self).__init__()
        self.conv_in = ConvGnReLU(
            n_features, n_features // 8, kernel_size=3, stride=2, padding=1)
        self.flat = nn.Flatten(1)
        self.linear1 = nn.Linear(n_features * 8 + 3, hidden_size)
        self.relu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, n_class)

    def forward(self, inputs, aux_feats):
        x = self.conv_in(inputs)
        x = self.flat(x)
        x = torch.concat([x, aux_feats], dim=1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, n_pose_bins=300, n_depth_bins=1000):
        super(ClassificationHead, self).__init__()

        self.n_depth_bins = n_depth_bins
        self.pose_clf = TaskHead(128 * 8, n_pose_bins)
        self.depth_clf = TaskHead(128 * 8, n_depth_bins)
        self.trans_clf = TaskHead(128 * 8, cfg.CELL_SIZE**2)

    def forward(self, features):
        pose_logits = self.pose_clf(features['clf_pool'], features['fov'])
        depth_logits = self.depth_clf(features['clf_pool'], features['fov'])
        trans_logits = self.trans_clf(features['clf_pool'], features['fov'])
        return pose_logits, depth_logits, trans_logits


class RegressionHead(nn.Module):
    def __init__(self, dim_pose=4, dim_trans=3):
        super(RegressionHead, self).__init__()
        self.pose_reg = TaskHead(256 * 8, dim_pose)
        self.depth_reg = TaskHead(256 * 8, 1)
        self.trans_reg = TaskHead(256 * 8, 2)

        nn.init.xavier_uniform_(self.depth_reg.linear2.weight,
                                gain=1.0 / cfg.Tz_BINS_NUM)
        nn.init.zeros_(self.depth_reg.linear2.bias)
        nn.init.xavier_uniform_(self.trans_reg.linear2.weight,
                                gain=1.0 / cfg.CELL_SIZE)
        nn.init.zeros_(self.trans_reg.linear2.bias)

    def forward(self, features):
        pose_res = self.pose_reg(features['reg_pool'], features['fov'])
        depth_res = self.depth_reg(features['reg_pool'], features['fov'])
        trans_res = self.trans_reg(features['reg_pool'], features['fov'])
        return pose_res, depth_res, trans_res


class MRCNet(nn.Module):
    def __init__(self, cfg):
        super(MRCNet, self).__init__()

        self.backbone = ResNet34_AsymUNet(
            n_decoders=cfg.n_decoders, track_running_stats=False,
            rgb_input_dim=4)
        self.decoder = PoseDecoder()
        self.classifier = ClassificationHead(
            n_pose_bins=cfg.n_pose_bin, n_depth_bins=cfg.n_depth_bin)
        self.regressor = RegressionHead(
            dim_pose=6 if cfg.use_6d else 4)
        self.registry = ObjectRegistry(cfg.dataset)
        self.renderer = rendering.Pytorch3DRenderer(
            cfg.input_img_size, cfg.input_img_size, self.registry)

        self.depth_min = cfg.depth_min
        self.depth_max = cfg.depth_max
        self.n_depth_bin = cfg.n_depth_bin
        self.n_pose_bin = cfg.n_pose_bin
        self.use_6d = cfg.use_6d

    def _make_predictions(self, mask_real, mask_synt, pose_logits,
                          depth_logits, trans_logits, pose_res,
                          depth_res, trans_res):
        depth = utils.depth_from_bin_logits(
            depth_logits, depth_res, self.depth_min, self.depth_max,
            self.n_depth_bin)
        trans_xy = utils.trans_from_bins(trans_logits, trans_res)
        trans_persp = torch.concat([trans_xy, depth], dim=-1)
        if self.use_6d:
            rot_ego = utils.rotation_from_prototypes(pose_logits, pose_res)
        else:
            quaternions = utils.quaternion_from_prototypes(
                pose_logits, pose_res)
            rot_ego = utils.quaternion_to_rotation(quaternions)
        predictions = {'roi_mask': mask_real,
                       'roi_mask_synt': mask_synt,
                       'roi_obj_R': rot_ego,
                       'quat_bin': pose_logits,
                       'quat_res': pose_res,
                       'depth_bin': depth_logits,
                       'depth_res': depth_res,
                       'trans_xy': trans_xy,
                       'trans_logits': trans_logits,
                       'trans_res': trans_res,
                       'translation': trans_persp}
        return predictions

    def _refine_step(self, x_clf, R_cur, t_cur, aux, grad_ckpt=False):
        """One synthetic-branch forward for the current pose estimate.

        Renders ``(R_cur, t_cur)``, runs the weight-shared backbone and decoder
        on the synthetic/real feature pair and predicts the residual to compose
        on top of the current pose.  ``R_cur``/``t_cur`` are treated as a
        detached base: only the residuals carry gradient within a step.
        """
        R_base = R_cur.detach()
        t_base = t_cur.detach()

        with torch.no_grad():
            out = self.renderer.render(
                aux['obj_cls'], R_base, t_base,
                [rendering.CameraView(K=aux['intrinsics'])])
            render = out['rgb'][:, 0]
            render_map = out['bbox_map'][:, 0]

        sync_inputs = torch.concat([render, render_map], dim=1)

        def synt_forward(sync_inputs, x_clf_in):
            x2, mask_synt = self.backbone(sync_inputs, aux['obj_cls'])
            x_synt_in = torch.concat(
                [x2, mask_synt.detach().sigmoid()], dim=1)
            x_reg = self.decoder(x_clf_in, x_synt_in)
            return x_reg, mask_synt

        if grad_ckpt:
            x_reg, mask_synt = torch.utils.checkpoint.checkpoint(
                synt_forward, sync_inputs, x_clf, use_reentrant=False)
        else:
            x_reg, mask_synt = synt_forward(sync_inputs, x_clf)

        x_reg['fov'] = torch.concat([
            t_base[..., :2] / t_base[..., -1:],
            aux['fov'][..., 2:]], dim=1)
        pose_res, depth_res, trans_res = self.regressor(x_reg)
        return {'render': render, 'render_map': render_map,
                'mask_synt': mask_synt, 'pose_res': pose_res,
                'depth_res': depth_res, 'trans_res': trans_res}

    def _compose_refine(self, step, R_cur, t_cur, aux, mask_real):
        """Compose one refinement residual onto the current pose estimate.

        ``R_k = ortho6d(pose_res) @ R_cur`` and the translation residual is
        added to the perspective coordinates / nearest depth bin of
        ``t_cur``, mirroring how ``_make_predictions`` composes iteration 0's
        residual onto the classification argmax.
        """
        image_size = (cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE)
        R_base = R_cur.detach()
        t_base = t_cur.detach()

        if self.use_6d:
            R_res = utils.compute_rotation_matrix_from_ortho6d(step['pose_res'])
            rot_ego = R_res @ R_base
        else:
            quat_res = utils.multiply_quaternion(
                step['pose_res'], utils.rotation_to_quaternion(R_base))
            rot_ego = utils.quaternion_to_rotation(quat_res)

        t_persp = utils.trans_3d_to_perspective(
            t_base, image_size, aux['intrinsics'])
        trans_2d_base = t_persp[..., :2]
        depth_base = t_persp[..., -1:]
        depth_id_base = utils.nearest_depth_bin(
            depth_base.squeeze(-1), self.depth_min, self.depth_max,
            self.n_depth_bin)
        depth_bins = utils.make_depth_bins(
            self.depth_min, self.depth_max, self.n_depth_bin,
            device=depth_base.device)
        depth_bins = torch.repeat_interleave(
            depth_bins, len(depth_id_base), dim=0)
        depth = torch.gather(
            depth_bins, dim=-1, index=depth_id_base.unsqueeze(-1)) \
            + step['depth_res']
        trans_xy = trans_2d_base + step['trans_res']
        translation = torch.concat([trans_xy, depth], dim=-1)

        return {'roi_mask': mask_real,
                'roi_mask_synt': step['mask_synt'],
                'roi_obj_R': rot_ego,
                'quat_res': step['pose_res'],
                'depth_res': step['depth_res'],
                'trans_xy': trans_xy,
                'trans_res': step['trans_res'],
                'translation': translation,
                'render': step['render'],
                'mask_synt': step['mask_synt'],
                'trans_2d_base': trans_2d_base,
                'depth_id_base': depth_id_base}

    def _refine_residual_mag(self, predictions):
        """Batch-level magnitude of the residual applied in one iteration.

        Relative rotation angle plus the depth residual normalized by the
        current depth.  A single scalar for the whole batch, so inference-only
        early stopping can never make the shared refinement path diverge
        per sample.
        """
        if self.use_6d:
            R_res = utils.compute_rotation_matrix_from_ortho6d(
                predictions['quat_res'])
            rot = utils.rotation_angle(R_res)
        else:
            rot = utils.angle_between_quaternion(
                predictions['quat_res'],
                torch.tensor(
                    [[1., 0., 0., 0.]],
                    device=predictions['quat_res'].device),
                is_degree=False)
        depth = predictions['translation'][..., -1:].abs().clamp(min=1e-6)
        trans = predictions['depth_res'].abs() / depth
        return float(torch.maximum(rot.max(), trans.max()))

    def forward(self, inputs, aux, targets=None, n_refine_iters=1,
                grad_ckpt=False, refine_stop_thresh=0.0):
        if n_refine_iters < 1:
            raise ValueError('n_refine_iters must be >= 1')
        weights = cfg.REFINE_ITER_WEIGHTS
        if weights is not None and len(weights) < n_refine_iters:
            raise ValueError(
                'REFINE_ITER_WEIGHTS has {} entries but '
                'n_refine_iters={}'.format(len(weights), n_refine_iters))

        # Classification bootstrap: unchanged, always iteration 0.
        x1, mask_real = self.backbone(inputs, aux['obj_cls'])
        x_real_in = torch.concat([x1, mask_real.detach().sigmoid()], dim=1)
        x_clf = self.decoder(x_real_in)
        x_clf['fov'] = aux['fov']
        pose_logits, depth_logits, trans_logits = self.classifier(x_clf)
        image_size = (cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE)

        with torch.no_grad():
            depth_id = torch.argmax(depth_logits, dim=-1)
            depth_bins = utils.make_depth_bins(
                self.depth_min, self.depth_max, self.n_depth_bin,
                device=depth_logits.device)
            depth_bins = torch.repeat_interleave(
                depth_bins, len(depth_id), dim=0)
            depth = torch.gather(
                depth_bins, dim=-1, index=depth_id.unsqueeze(-1))

            zero_res = torch.zeros(len(trans_logits), 2).to(trans_logits)
            trans_xy = utils.trans_from_bins(trans_logits, zero_res)
            trans_persp = torch.cat([trans_xy, depth], dim=-1)
            trans_3d = utils.perspective_to_trans_3d(
                trans_persp, image_size, aux['intrinsics'])

            pose_id = torch.argmax(pose_logits, dim=-1)
            quat_init = utils.quaternion_from_bin_logits(F.one_hot(
                pose_id, num_classes=self.n_pose_bin))
            rot_init = utils.quaternion_to_rotation(quat_init)

        # Iteration 0 bootstraps from the classification argmax; later
        # iterations re-render and refine the previous estimate.
        R_cur, t_cur = rot_init, trans_3d
        predictions = None
        losses = {}
        refine_iters_used = n_refine_iters
        residual_mag = None
        for k in range(n_refine_iters):
            step = self._refine_step(
                x_clf, R_cur, t_cur, aux, grad_ckpt=grad_ckpt)
            if k == 0:
                predictions = self._make_predictions(
                    mask_real, step['mask_synt'], pose_logits, depth_logits,
                    trans_logits, step['pose_res'], step['depth_res'],
                    step['trans_res'])
                predictions['render'] = step['render']
                predictions['mask_synt'] = step['mask_synt']
                trans_2d_base = trans_xy
                depth_id_base = depth_id
                if targets:
                    targets['quat_init'] = quat_init
                    targets['trans_2d'] = trans_2d_base
                    targets['depth_id'] = depth_id_base
                    iter_losses = self._compute_loss(predictions, targets)
            else:
                predictions = self._compose_refine(
                    step, R_cur, t_cur, aux, mask_real)
                predictions['quat_bin'] = pose_logits
                predictions['depth_bin'] = depth_logits
                predictions['trans_logits'] = trans_logits
                trans_2d_base = predictions.pop('trans_2d_base')
                depth_id_base = predictions.pop('depth_id_base')
                if targets:
                    iter_losses = self._compute_refine_loss(
                        predictions, targets, R_cur, t_cur, trans_2d_base,
                        depth_id_base)

            if targets:
                weight = 1.0 if weights is None else float(weights[k])
                for name, value in iter_losses.items():
                    if weight != 1.0:
                        value = value * weight
                    key = name if k == 0 else 'iter{}/{}'.format(k, name)
                    losses[key] = value

            if k < n_refine_iters - 1:
                R_cur = predictions['roi_obj_R']
                t_cur = utils.perspective_to_trans_3d(
                    predictions['translation'], image_size, aux['intrinsics'])
                if refine_stop_thresh > 0:
                    residual_mag = self._refine_residual_mag(predictions)
                    if residual_mag < refine_stop_thresh:
                        refine_iters_used = k + 1
                        break

        if targets:
            predictions['losses'] = losses
        if n_refine_iters > 1:
            predictions['refine_iters_used'] = refine_iters_used
            predictions['refine_residual_mag'] = residual_mag
        return predictions

    def _compute_refine_loss(self, predictions, targets, R_cur, t_cur,
                             trans_2d_base, depth_id_base):
        """Deep supervision of a refinement iteration's residual.

        The GT symmetry group is re-matched against this iteration's pose
        (same disambiguation as ``_compute_loss``) and only the regression
        terms are returned, since the classification heads are not re-run.
        """
        image_size = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)  # constant

        quat_gt = utils.rotation_to_quaternion(targets['roi_obj_R'])
        trans_gt = targets['roi_obj_t']
        quat_pred = utils.rotation_to_quaternion(predictions['roi_obj_R'])
        trans_pred = utils.perspective_to_trans_3d(
            predictions['translation'], image_size, targets['roi_camK'])
        cad = self.registry.gather(targets['obj_cls'])
        quat_gt, trans_gt = utils.top_matched_pose(
            quat_gt, trans_gt, quat_pred, trans_pred,
            cad['vertices_correlation'],
            cad['quaternion_symmetries'],
            cad['translation_symmetries'], cad['symmetries_mask'])

        loss_dict = utils.vertex_rotation_reg_loss(
            predictions['quat_res'], quat_gt, R_cur.detach(),
            cad['vertices'], cad['vertices_mask'],
            image_size, targets['roi_camK'], self.use_6d)
        loss_dict.update(utils.vertex_translation_reg_loss(
            predictions['trans_res'], trans_2d_base, trans_gt,
            predictions['depth_res'], depth_id_base, targets['roi_camK'],
            image_size, self.depth_min, self.depth_max, self.n_depth_bin))
        return loss_dict

    def _compute_loss(self, predictions, targets):
        assert 'roi_obj_R' in targets

        loss_dict = {}
        image_size = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)  # constant

        quat_gt = utils.rotation_to_quaternion(targets['roi_obj_R'])
        trans_gt = targets['roi_obj_t']
        quat_pred = utils.rotation_to_quaternion(predictions['roi_obj_R'])
        trans_pred = utils.perspective_to_trans_3d(
            predictions['translation'], image_size, targets['roi_camK'])
        cad = self.registry.gather(targets['obj_cls'])
        quat_gt, trans_gt = utils.top_matched_pose(
            quat_gt, trans_gt, quat_pred, trans_pred,
            cad['vertices_correlation'],
            cad['quaternion_symmetries'],
            cad['translation_symmetries'], cad['symmetries_mask'])

        loss_dict = utils.vertex_rotation_loss(
            predictions['quat_bin'], predictions['quat_res'],
            quat_gt, targets['quat_bin'], targets['quat_init'],
            cad['vertices'], cad['vertices_mask'],
            image_size, targets['roi_camK'], self.use_6d)

        loss_trans_dict = utils.vertex_translation_loss(
            predictions['trans_logits'], predictions['trans_res'],
            targets['trans_2d'], trans_gt, predictions['depth_bin'],
            predictions['depth_res'], targets['depth_id'], targets['roi_camK'],
            image_size,
            self.depth_min, self.depth_max, self.n_depth_bin)
        loss_dict.update(loss_trans_dict)

        bce_fn = nn.BCEWithLogitsLoss()
        mask_gt = targets['roi_mask'].unsqueeze(1)
        loss_dict['mask_visib'] = cfg.WEIGHT_MASK * bce_fn(
            predictions['roi_mask'], mask_gt)
        return loss_dict
