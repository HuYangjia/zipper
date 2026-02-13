import torch

from .._C import ops
# from ..utils import ceil_divide


# def quantize_w4a4_act_fuse_sparse_mult(
#     input: torch.Tensor,
#     output: torch.Tensor | None = None,
#     oscales: torch.Tensor | None = None,
#     lora_down: torch.Tensor | None = None,
#     lora_act_out: torch.Tensor | None = None,
#     smooth: torch.Tensor | None = None,
#     fuse_glu: bool = False,
#     fp4: bool = False,
#     pad_size: int = 256,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Quantizes activations and computes LoRA down-projection using SVDQuant W4A4 CUDA kernel.

#     Parameters
#     ----------
#     input : torch.Tensor, shape (M, K), dtype bfloat16/float16
#         Input activations.
#     output : torch.Tensor or None, shape (M_pad, K // 2), dtype uint8, optional
#         Packed output tensor for quantized activations. Allocated if None.
#     oscales : torch.Tensor or None, shape (K // G, M_pad), dtype float8_e4m3fn for NVFP4 or input dtype for INT4, optional
#         Output scales tensor. Allocated if None.
#     lora_down : torch.Tensor or None, shape (K, R), dtype bfloat16/float16, optional
#         Packed LoRA down-projection weights.
#     lora_act_out : torch.Tensor or None, shape (M_pad, R), dtype float32, optional
#         Packed output tensor for LoRA activations. Allocated if None.
#     smooth : torch.Tensor or None, optional, dtype bfloat16/float16
#         Smoothing factor for quantization.
#     fuse_glu : bool, default=False
#         If True, fuse GLU activation.
#     fp4 : bool, default=False
#         If True, use NVFP4 quantization; else INT4.
#     pad_size : int, default=256
#         Pad batch size to a multiple of this value for efficient CUDA execution.

#     Returns
#     -------
#     output : torch.Tensor, shape (M_pad, K // 2), dtype uint8
#         Packed quantized activations.
#     oscales : torch.Tensor, shape (K // G, M_pad), dtype float8_e4m3fn for NVFP4 or input dtype for INT4
#         Output scales.
#     lora_act_out : torch.Tensor, shape (M_pad, R), dtype float32
#         Packed LoRA activation output.

#     Notes
#     -----
#     Notations:

#     - M: batch size
#     - K: input channels
#     - R: LoRA rank
#     - G: group size (64 for INT4, 16 for NVFP4)
#     - M_pad: padded batch size = ceil(M / pad_size) * pad_size
#     """
#     batch_size, channels = input.shape
#     rank = lora_down.shape[1]
#     batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size
#     if output is None:
#         output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
#     if oscales is None:
#         if fp4:
#             assert channels % 16 == 0
#             oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
#         else:
#             assert channels % 64 == 0
#             oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
#     if lora_act_out is None:
#         lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

#     ops.quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)
#     return output, oscales, lora_act_out



def ceil_divide(a: int, b: int) -> int:
    return (a + b - 1) // b


def quantize_w4a4_act(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    pad_size: int = 256,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Python wrapper for C++ op: ops.quantize_w4a4_act(input, output, oscales)

    Parameters
    ----------
    input : torch.Tensor, shape (M, K), dtype float16/bfloat16, CUDA
    output : torch.Tensor or None, shape (M_pad, K//2), dtype uint8, CUDA
        Packed INT4 activations (2 values per byte). Allocated if None.
    oscales : torch.Tensor or None, shape (K//G, M_pad), dtype same as input, CUDA
        Per-group scales. Allocated if None.
    pad_size : int
        Pad M up to multiple of pad_size.
    group_size : int
        G in K//G. Commonly 64 for INT4.

    Returns
    -------
    output, oscales
    """

    if input is None:
        raise ValueError("quantize_w4a4_act: input must be provided")
    if input.ndim != 2:
        raise ValueError(f"quantize_w4a4_act: input must be 2D (M,K), got {tuple(input.shape)}")
    if not input.is_cuda:
        raise ValueError("quantize_w4a4_act: input must be a CUDA tensor")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"quantize_w4a4_act: input must be fp16/bf16, got {input.dtype}")

    M, K = input.shape
    if K % 2 != 0:
        raise ValueError(f"quantize_w4a4_act: K must be even for int4 packing (K//2 bytes), got K={K}")
    if K % group_size != 0:
        raise ValueError(f"quantize_w4a4_act: K must be divisible by group_size={group_size}, got K={K}")

    M_pad = ceil_divide(M, pad_size) * pad_size
    dev = input.device

    # Ensure contiguous for kernel-side contig checks
    input_c = input.contiguous()

    # Allocate output if needed
    if output is None:
        output = torch.empty((M_pad, K // 2), device=dev, dtype=torch.uint8)
    else:
        if not output.is_cuda or output.device != dev:
            raise ValueError("quantize_w4a4_act: output must be CUDA tensor on same device as input")
        if output.dtype != torch.uint8:
            raise TypeError(f"quantize_w4a4_act: output must be uint8, got {output.dtype}")
        if tuple(output.shape) != (M_pad, K // 2):
            raise ValueError(f"quantize_w4a4_act: output shape must be {(M_pad, K//2)}, got {tuple(output.shape)}")
        output = output.contiguous()

    # Allocate oscales if needed
    if oscales is None:
        oscales = torch.empty((K // group_size, M_pad), device=dev, dtype=input.dtype)
    else:
        if not oscales.is_cuda or oscales.device != dev:
            raise ValueError("quantize_w4a4_act: oscales must be CUDA tensor on same device as input")
        if oscales.dtype != input.dtype:
            raise TypeError(f"quantize_w4a4_act: oscales dtype must match input ({input.dtype}), got {oscales.dtype}")
        if tuple(oscales.shape) != (K // group_size, M_pad):
            raise ValueError(
                f"quantize_w4a4_act: oscales shape must be {(K//group_size, M_pad)}, got {tuple(oscales.shape)}"
            )
        oscales = oscales.contiguous()

    # Call C++ op
    ops.quantize_w4a4_act(input_c, output, oscales)
    return output, oscales