# zipper/ops/quantize_bench.py
import argparse
import sys
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from zipper.ops import quantize_w4a4_act

def bench_quantize(N: int, K: int, warmup: int, iters: int, seed: int):
    assert torch.cuda.is_available()
    device = "cuda"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # input: [N, K] FP16
    x = torch.randn((N, K), device=device, dtype=torch.float16)

    # warmup
    for _ in range(warmup):
        out, oscales = quantize_w4a4_act(x)
    torch.cuda.synchronize()

    # timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(iters):
        start.record()
        out, oscales = quantize_w4a4_act(x)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times = torch.tensor(times_ms, device="cpu")
    avg = float(times.mean())
    p50 = float(times.median())
    p90 = float(times.kthvalue(max(1, int(0.90 * len(times))))[0])
    p99 = float(times.kthvalue(max(1, int(0.99 * len(times))))[0])

    # 量化“吞吐”可以粗略按读取 input bytes 来算（这里只给一个简单参考）
    # bytes_in = N*K*2 (fp16)
    bytes_in = N * K * 2
    gbps_in = (bytes_in / (avg / 1e3)) / 1e9

    print(f"[quantize_w4a4_act] N={N} K={K} dtype=fp16")
    print(f"  time ms: avg={avg:.4f}  p50={p50:.4f}  p90={p90:.4f}  p99={p99:.4f}  (iters={iters}, warmup={warmup})")
    print(f"  approx input GB/s: {gbps_in:.3f}")
    return out, oscales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True, help="batch size (rows)")
    ap.add_argument("--K", type=int, required=True, help="channels (cols)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # int4 packing typically requires K even; keep it consistent with wrapper checks
    if args.K % 2 != 0:
        raise ValueError(f"K must be even for int4 packing, got K={args.K}")

    bench_quantize(args.N, args.K, args.warmup, args.iters, args.seed)


if __name__ == "__main__":
    main()
