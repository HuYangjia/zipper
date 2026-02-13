import torch
from .._C import ops


def spmm_csr_fp16_wxt(
    X: torch.Tensor,
    csrOffsets: torch.Tensor,
    csrCols: torch.Tensor,
    csrValsFp16: torch.Tensor,
    Y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calls C++/CUDA: nunchaku::kernels::spmm_csr_fp16_wxt

    Semantics (based on your C++ shape checks):
      - X: [N, K], FP16, CUDA, row-major contiguous
      - CSR represents W: [M, K]
          csrOffsets: [M+1] int32
          csrCols:    [nnz] int32
          csrValsFp16:[nnz] FP16
      - Y: [M, N], FP16 (allocated if None)

    Computes: Y = W @ X^T   (so output is [M, N])
    """

    # ---- required inputs must be present ----
    if X is None or csrOffsets is None or csrCols is None or csrValsFp16 is None:
        raise ValueError("spmm_csr_fp16_wxt: X/csrOffsets/csrCols/csrValsFp16 must all be provided")

    # ---- basic shape checks ----
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [N,K], got shape={tuple(X.shape)}")
    if csrOffsets.ndim != 1:
        raise ValueError(f"csrOffsets must be 1D [M+1], got shape={tuple(csrOffsets.shape)}")
    if csrCols.ndim != 1:
        raise ValueError(f"csrCols must be 1D [nnz], got shape={tuple(csrCols.shape)}")
    if csrValsFp16.ndim != 1:
        raise ValueError(f"csrValsFp16 must be 1D [nnz], got shape={tuple(csrValsFp16.shape)}")

    N, K = X.shape
    M = int(csrOffsets.numel()) - 1
    if M <= 0:
        raise ValueError("csrOffsets length must be M+1 with M>0")

    nnz = int(csrValsFp16.numel())
    if int(csrCols.numel()) != nnz:
        raise ValueError(f"csrCols.numel() must equal nnz; got {csrCols.numel()} vs {nnz}")

    # ---- device checks (your C++ requires CUDA for all tensors) ----
    if not X.is_cuda:
        raise ValueError("X must be a CUDA tensor")
    dev = X.device
    for t, name in [(csrOffsets, "csrOffsets"), (csrCols, "csrCols"), (csrValsFp16, "csrValsFp16")]:
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != dev:
            raise ValueError(f"{name} must be on same device as X (got {t.device} vs {dev})")

    # ---- dtype checks / conversions to match your C++ check_dtype ----
    if X.dtype != torch.float16:
        raise TypeError(f"X must be torch.float16, got {X.dtype}")
    if csrValsFp16.dtype != torch.float16:
        raise TypeError(f"csrValsFp16 must be torch.float16, got {csrValsFp16.dtype}")

    # Prefer enforcing int32 to avoid silent extra copies later; convert if needed.
    if csrOffsets.dtype != torch.int32:
        csrOffsets = csrOffsets.to(dtype=torch.int32)
    if csrCols.dtype != torch.int32:
        csrCols = csrCols.to(dtype=torch.int32)

    # ---- contiguity (your C++ requires cuda_contig; X/Y also enforce stride(1)==1) ----
    X = X.contiguous()
    csrOffsets = csrOffsets.contiguous()
    csrCols = csrCols.contiguous()
    csrValsFp16 = csrValsFp16.contiguous()

    # ---- output allocation / validation ----
    if Y is None:
        Y = torch.empty((M, N), device=dev, dtype=torch.float16)
    else:
        if not Y.is_cuda or Y.device != dev:
            raise ValueError("Y must be a CUDA tensor on the same device as X")
        if Y.dtype != torch.float16:
            raise TypeError(f"Y must be torch.float16, got {Y.dtype}")
        if tuple(Y.shape) != (M, N):
            raise ValueError(f"Y must have shape [M,N]=[{M},{N}], got {tuple(Y.shape)}")
        Y = Y.contiguous()

    # ---- call into C++ op (optional args supported by your binding, but we pass all explicitly) ----
    ops.spmm_csr_fp16_wxt(X, csrOffsets, csrCols, csrValsFp16, Y)
    return Y
