import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ==============================================================================
# TRITON JIT KERNELS
# ==============================================================================
if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_HW': 64,  'BLOCK_C': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_HW': 64,  'BLOCK_C': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 32}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 32}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 64}, num_warps=8, num_stages=2),
        ],
        key=['B', 'C', 'H', 'W', 'P'],
    )
    @triton.jit
    def _corr_spatial_matmul_kernel(
            x1_ptr, x2_ptr, out_ptr,
            B, C, H, W, P,
            stride_b1, stride_c1, stride_h1, stride_w1,
            stride_b2, stride_c2, stride_h2, stride_w2,
            stride_ob, stride_oph, stride_opw, stride_oh, stride_ow,
            BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        b_phw = tl.program_id(0)
        b = b_phw // (P * P)
        phw = b_phw % (P * P)

        ph = phw // P
        pw = phw % P

        dh = ph - P // 2
        dw = pw - P // 2

        spatial_block_idx = tl.program_id(1) * BLOCK_HW
        offs_hw = spatial_block_idx + tl.arange(0, BLOCK_HW)

        h = offs_hw // W
        w = offs_hw % W
        mask_x1 = offs_hw < (H * W)

        h2 = h + dh
        w2 = w + dw
        mask_x2 = mask_x1 & (h2 >= 0) & (h2 < H) & (w2 >= 0) & (w2 < W)

        h2_clamped = tl.where(mask_x2, h2, 0)
        w2_clamped = tl.where(mask_x2, w2, 0)

        x1_spatial_bytes = h * stride_h1 + w * stride_w1
        x2_spatial_bytes = h2_clamped * stride_h2 + w2_clamped * stride_w2

        x1_batch_ptr = x1_ptr + b * stride_b1
        x2_batch_ptr = x2_ptr + b * stride_b2

        acc = tl.zeros([BLOCK_HW, BLOCK_C], dtype=tl.float32)

        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C

            ptr_x1 = x1_batch_ptr + x1_spatial_bytes[:, None] + offs_c[None, :] * stride_c1
            ptr_x2 = x2_batch_ptr + x2_spatial_bytes[:, None] + offs_c[None, :] * stride_c2

            v1 = tl.load(ptr_x1, mask=mask_x1[:, None] & mask_c[None, :], other=0.0)
            v2 = tl.load(ptr_x2, mask=mask_x2[:, None] & mask_c[None, :], other=0.0)

            acc += v1.to(tl.float32) * v2.to(tl.float32)

        out_values = tl.sum(acc, axis=1)

        out_offset = (
                b * stride_ob +
                ph * stride_oph +
                pw * stride_opw +
                h * stride_oh +
                w * stride_ow
        )

        tl.store(out_ptr + out_offset, out_values, mask=mask_x1)


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_HW': 32,  'BLOCK_C': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_HW': 32,  'BLOCK_C': 64}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_HW': 64,  'BLOCK_C': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_HW': 64,  'BLOCK_C': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 64,  'BLOCK_C': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 64}, num_warps=8, num_stages=2),
        ],
        key=['B', 'C', 'H', 'W', 'P'],
    )
    @triton.jit
    def _corr_spatial_matmul_bwd_kernel(
            grad_out_ptr, x1_ptr, x2_ptr, grad_x1_ptr, grad_x2_ptr,
            B, C, H, W, P,
            stride_gob, stride_goph, stride_gopw, stride_goh, stride_gow,
            stride_b1, stride_c1, stride_h1, stride_w1,
            stride_b2, stride_c2, stride_h2, stride_w2,
            stride_gx1b, stride_gx1c, stride_gx1h, stride_gx1w,
            stride_gx2b, stride_gx2c, stride_gx2h, stride_gx2w,
            BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_hw = tl.program_id(0)
        pid_c = tl.program_id(1)
        b = tl.program_id(2)

        offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
        h = offs_hw // W
        w = offs_hw % W
        mask_hw = offs_hw < (H * W)

        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask_tile = mask_hw[:, None] & mask_c[None, :]

        go_batch_ptr = grad_out_ptr + b * stride_gob
        x1_batch_ptr = x1_ptr + b * stride_b1
        x2_batch_ptr = x2_ptr + b * stride_b2

        acc_grad_x1 = tl.zeros([BLOCK_HW, BLOCK_C], dtype=tl.float32)
        acc_grad_x2 = tl.zeros([BLOCK_HW, BLOCK_C], dtype=tl.float32)

        x1_spatial_offset = h * stride_h1 + w * stride_w1
        ptr_x1 = x1_batch_ptr + x1_spatial_offset[:, None] + offs_c[None, :] * stride_c1
        v1 = tl.load(ptr_x1, mask=mask_tile, other=0.0)

        go_spatial_base = h * stride_goh + w * stride_gow

        for ph in range(P):
            dh = ph - P // 2
            h2 = h + dh
            mask_h2_valid = (h2 >= 0) & (h2 < H)

            for pw in range(P):
                dw = pw - P // 2
                w2 = w + dw
                mask_w2_valid = (w2 >= 0) & (w2 < W)
                mask_x2 = mask_hw & mask_h2_valid & mask_w2_valid

                go_offset = ph * stride_goph + pw * stride_gopw + go_spatial_base
                ptr_go = go_batch_ptr + go_offset[:, None]
                v_go = tl.load(ptr_go, mask=mask_hw[:, None], other=0.0).to(tl.float32)

                h2_clamped = tl.where(mask_x2, h2, 0)
                w2_clamped = tl.where(mask_x2, w2, 0)
                x2_spatial_offset = h2_clamped * stride_h2 + w2_clamped * stride_w2
                ptr_x2 = x2_batch_ptr + x2_spatial_offset[:, None] + offs_c[None, :] * stride_c2
                v2 = tl.load(ptr_x2, mask=mask_x2[:, None] & mask_c[None, :], other=0.0)

                grad_x1_step = v_go * v2.to(tl.float32)
                acc_grad_x1 += tl.where(mask_x2[:, None], grad_x1_step, 0.0)

                h_inv = h - dh
                w_inv = w - dw
                mask_inv = mask_hw & (h_inv >= 0) & (h_inv < H) & (w_inv >= 0) & (w_inv < W)

                h_inv_clamped = tl.where(mask_inv, h_inv, 0)
                w_inv_clamped = tl.where(mask_inv, w_inv, 0)

                go_inv_offset = ph * stride_goph + pw * stride_gopw + (
                            h_inv_clamped * stride_goh + w_inv_clamped * stride_gow)
                ptr_go_inv = go_batch_ptr + go_inv_offset[:, None]
                v_go_inv = tl.load(ptr_go_inv, mask=mask_inv[:, None], other=0.0).to(tl.float32)

                x1_inv_spatial_offset = h_inv_clamped * stride_h1 + w_inv_clamped * stride_w1
                ptr_x1_inv = x1_batch_ptr + x1_inv_spatial_offset[:, None] + offs_c[None, :] * stride_c1
                v1_inv = tl.load(ptr_x1_inv, mask=mask_inv[:, None] & mask_c[None, :], other=0.0)

                grad_x2_step = v_go_inv * v1_inv.to(tl.float32)
                acc_grad_x2 += tl.where(mask_inv[:, None], grad_x2_step, 0.0)

        grad_x1_batch_ptr = grad_x1_ptr + b * stride_gx1b
        grad_x2_batch_ptr = grad_x2_ptr + b * stride_gx2b

        ptr_gx1 = grad_x1_batch_ptr + x1_spatial_offset[:, None] + offs_c[None, :] * stride_gx1c
        tl.store(ptr_gx1, acc_grad_x1.to(grad_x1_ptr.dtype.element_ty), mask=mask_tile)

        ptr_gx2 = grad_x2_batch_ptr + x1_spatial_offset[:, None] + offs_c[None, :] * stride_gx2c
        tl.store(ptr_gx2, acc_grad_x2.to(grad_x2_ptr.dtype.element_ty), mask=mask_tile)


# ==============================================================================
# AUTOGRAD WRAPPER
# ==============================================================================
class _CorrTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, P):
        if not HAS_TRITON:
            raise RuntimeError("Triton installation not detected on this system setup.")

        x1 = x1.contiguous()
        x2 = x2.contiguous()
        B, C, H, W = x1.shape

        out = torch.empty(B, P, P, H, W, device=x1.device, dtype=x1.dtype)

        grid = lambda meta: (
            B * P * P,
            triton.cdiv(H * W, meta['BLOCK_HW'])
        )

        _corr_spatial_matmul_kernel[grid](
            x1, x2, out,
            B, C, H, W, P,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        )

        ctx.save_for_backward(x1, x2)
        ctx.P = P
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x1, x2 = ctx.saved_tensors
        P = ctx.P
        B, C, H, W = x1.shape

        grad_out = grad_out.contiguous()

        grad_x1 = torch.empty_like(x1)
        grad_x2 = torch.empty_like(x2)

        grid = lambda meta: (
            triton.cdiv(H * W, meta['BLOCK_HW']),
            triton.cdiv(C, meta['BLOCK_C']),
            B
        )

        _corr_spatial_matmul_bwd_kernel[grid](
            grad_out, x1, x2, grad_x1, grad_x2,
            B, C, H, W, P,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3), grad_out.stride(4),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            grad_x1.stride(0), grad_x1.stride(1), grad_x1.stride(2), grad_x1.stride(3),
            grad_x2.stride(0), grad_x2.stride(1), grad_x2.stride(2), grad_x2.stride(3),
        )

        return grad_x1, grad_x2, None


# ==============================================================================
# FUNCTIONAL IMPLEMENTATIONS
# ==============================================================================
def corr_slice(x1: torch.Tensor, x2: torch.Tensor, P: int) -> torch.Tensor:
    """Zero-unfold spatial correlation engine.

    Memory efficient view slicing strategy. Fully compatible with ONNX tracing.
    """
    B, C, H, W = x1.shape
    pad = P // 2

    x2_padded = torch.nn.functional.pad(x2, (pad, pad, pad, pad))
    out = torch.zeros((B, P, P, H, W), dtype=x1.dtype, device=x1.device)

    for i in range(P):
        for j in range(P):
            x2_slice = x2_padded[:, :, i: i + H, j: j + W]
            sim = torch.sum(x1 * x2_slice, dim=1)
            out[:, i, j, :, :] = sim

    return out


def corr_triton(x1: torch.Tensor, x2: torch.Tensor, P: int) -> torch.Tensor:
    """Triton-accelerated block-fused spatial correlation engine."""
    return _CorrTritonFn.apply(x1, x2, P)


# ==============================================================================
# NN.MODULE INTERFACE
# ==============================================================================
class SpatialCorrelationSampler(nn.Module):
    """Spatial Correlation Sampler module supporting fused Triton block processing
    and memory-efficient slicing fallback paths.

    Args:
        patch_size (int): Size of neighborhood search window (must be odd).
        backend (str): Computational engine selection ('triton', 'slice', or 'auto').
    """

    def __init__(self, patch_size: int, backend: str = "auto") -> None:
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be an odd integer, received: {patch_size}")

        valid_backends = {"auto", "triton", "slice"}
        if backend not in valid_backends:
            raise ValueError(f"Backend must be one of {valid_backends}, received: {backend}")

        self.patch_size = patch_size
        self.backend = backend

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return corr_slice(x1, x2, self.patch_size)
        if self.backend == "triton":
            return corr_triton(x1, x2, self.patch_size)
        elif self.backend == "slice":
            return corr_slice(x1, x2, self.patch_size)

        # Automatic Backend Selection
        if HAS_TRITON and x1.is_cuda:
            return corr_triton(x1, x2, self.patch_size)
        return corr_slice(x1, x2, self.patch_size)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"patch_size={self.patch_size}, "
                f"backend='{self.backend}', "
                f"triton_available={HAS_TRITON})")