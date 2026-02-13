import argparse
import time
import sys
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from zipper.ops import spmm_csr_fp16_wxt

def make_int4_csr_weight(M: int, K: int, density: float, seed: int = 0, device="cuda"):
    """
    Build a CSR weight matrix W of shape [M, K]:
, where:
      - col indices int32
      - offsets int32
      - values stored as FP16, but generated via INT4 quant -> dequant to FP16
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # 每行 nnz 数（固定密度，方便控制）
    nnz_per_row = max(1, int(round(K * density)))
    nnz = M * nnz_per_row

    # csrOffsets: [M+1]
    csrOffsets = torch.arange(0, nnz + 1, step=nnz_per_row, device=device, dtype=torch.int32)

    # csrCols: [nnz], 每行随机不重复列
    # 用 topk trick 生成不重复列：对每行生成 K 个随机数取 topk 的 indices
    #（K 很大时会慢；如果你 K 可能很大，可以换更轻的采样方式）
    rand = torch.rand((M, K), device=device, generator=g)
    cols = torch.topk(rand, k=nnz_per_row, dim=1, largest=True).indices.to(torch.int32)  # [M, nnz_per_row]
    csrCols = cols.reshape(-1).contiguous()

    # 生成“原始”权重值，再做 int4 量化与反量化，最终存 FP16
    w_fp = torch.randn((nnz,), device=device, generator=g, dtype=torch.float32)

    # 简单 per-tensor scale 的 int4 对称量化：q = clamp(round(w/scale), -8, 7)
    # scale 取 max/7，保证覆盖范围
    max_abs = w_fp.abs().max().clamp(min=1e-6)
    scale = (max_abs / 7.0).item()
    q = torch.clamp(torch.round(w_fp / scale), -8, 7)

    w_deq = (q * scale).to(torch.float16).contiguous()  # csrValsFp16

    return csrOffsets, csrCols, w_deq


def bench_spmm(M: int, K: int, N: int, density: float, warmup: int, iters: int, seed: int):
    assert torch.cuda.is_available()
    device = "cuda"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # X: [N, K] FP16 (match your C++ checks)
    X = torch.randn((N, K), device=device, dtype=torch.float16)

    # CSR W: [M, K] as (offsets int32, cols int32, vals fp16)
    csrOffsets, csrCols, csrValsFp16 = make_int4_csr_weight(M, K, density, seed=seed, device=device)

    # Output Y: [M, N] FP16
    Y = torch.empty((M, N), device=device, dtype=torch.float16)

    # 预热（避免首次 kernel / lazy init / cudnn/cusparse setup 影响）
    for _ in range(warmup):
        spmm_csr_fp16_wxt(X, csrOffsets, csrCols, csrValsFp16, Y)
    torch.cuda.synchronize()

    # CUDA event 计时（推荐，比 time.time 准）
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(iters):
        start.record()
        spmm_csr_fp16_wxt(X, csrOffsets, csrCols, csrValsFp16, Y)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times = torch.tensor(times_ms, device="cpu")
    avg = float(times.mean())
    p50 = float(times.median())
    p90 = float(times.kthvalue(max(1, int(0.90 * len(times))))[0])
    p99 = float(times.kthvalue(max(1, int(0.99 * len(times))))[0])

    # 简单算一下理论 FLOPs：SpMM (M x K sparse) * (K x N dense) ~ 2 * nnz * N
    nnz = int(csrValsFp16.numel())
    flops = 2.0 * nnz * N
    tflops = (flops / (avg / 1e3)) / 1e12  # avg ms -> s

    print(f"[spmm_csr_fp16_wxt] M={M} K={K} N={N} density={density} nnz={nnz} nnz/row={nnz//M}")
    print(f"  time ms: avg={avg:.4f}  p50={p50:.4f}  p90={p90:.4f}  p99={p99:.4f}  (iters={iters}, warmup={warmup})")
    print(f"  approx TFLOPS (2*nnz*N): {tflops:.3f}")
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--density", type=float, default=0.01, help="nnz density per row, e.g., 0.01 means 1% nonzeros")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bench_spmm(args.M, args.K, args.N, args.density, args.warmup, args.iters, args.seed)


if __name__ == "__main__":
    main()

