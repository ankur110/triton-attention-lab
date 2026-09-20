# Flash Attention 2 in Triton

A FlashAttention-2 forward kernel written from scratch in Triton, with grouped-query
attention (GQA) and causal masking, benchmarked against PyTorch SDPA's FLASH backend
and a naive eager implementation.

All numbers below are from an A100-SXM4-40GB (torch 2.8.0+cu128, triton 3.4.0),
fp16, `H=32` query heads, `H_kv=8` KV heads, `head_dim=64`, batch 1, causal.

## Layout

| Path | What it is |
| --- | --- |
| `fa2_fwd.py` | The kernel. `_fa2_fwd` (the `@triton.jit` kernel) and `fa2_fwd` (the launcher). |
| `fa2_mod.py` | A copy of `fa2_fwd.py`, written by the notebook so subprocesses can import the kernel by module name. Do not edit it directly. |
| `fa_bench_notebook.ipynb` | Correctness checks, the block/warp/stage sweep, the head-to-head benchmark, and the bandwidth analysis. |
| `model/llama3_1b.py` | Llama-3.2-1B from scratch (GQA, RoPE, KV cache). The consumer the kernel is aimed at. |
| `*.csv` | Committed results from the notebook runs above, so the numbers are readable without a GPU. Re-running the notebook overwrites them. |

## The kernel

`fa2_fwd(q, k, v, causal=False, sm_scale=None, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)`

- `q: (B, H, S_q, d)`, `k, v: (B, H_kv, S_kv, d)` — any `H % H_kv == 0` grouping.
- Returns `o: (B, H, S_q, d)` in q's dtype and `lse: (B, H, S_q)` in fp32.
- `d` must be a power of two and `>= 16`. Ragged `S` is handled by masking, so
  `S` need not be a multiple of `BLOCK_M`.
- Causal currently requires `S_q == S_kv` (the mask is aligned to the diagonal of a
  square score matrix).

One program handles one `BLOCK_M` block of query rows for one `(batch, head)` pair.
It runs in two loops: fully-visible key blocks with only a bounds mask, then the
diagonal blocks that also need the causal mask. Splitting them keeps the `tl.where`
off the majority of blocks. GQA is a pointer trick — `kv_h = h // N_REP`, so the four
query heads in a group read the same K/V without materializing a `repeat_interleave`
copy.

`lse` is returned because it is what a backward pass needs to recompute softmax.

## Results

Forward pass, `p50` of `triton.testing.do_bench`, causal:

| S | eager | SDPA flash | ours (64/64/4/2) | ours vs flash |
| --- | --- | --- | --- | --- |
| 1024 | 623.6 us | 77.8 us | **69.6 us** (61.7 TFLOP/s) | **1.12x** |
| 4096 | 9561 us | 480.3 us | 629.8 us (109.1 TFLOP/s) | 0.76x |
| 8192 | 40495 us | 1682.9 us | 2276.9 us (120.7 TFLOP/s) | 0.74x |

So: comfortably ahead of eager everywhere, ahead of cuDNN/CUTLASS flash at short
sequences, and ~25% behind it once the problem is large enough for their
warp-specialized pipelining and register-level tiling to pay off.

A 60-point sweep over `BLOCK_M ∈ {64, 128}`, `BLOCK_N ∈ {32, 64, 128}`,
`num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3}` picks `64/64/4/2` at every sequence
length, and `triton.autotune` over the same space finds nothing better — within
noise of the hand-picked default. The sweep is in `sweep_configs.csv`.

Correctness is checked against a float64 reference computed a few heads at a time,
with the tolerance set to SDPA-flash's own error against that reference rather than a
fixed epsilon, so the kernel is held to "no worse than flash" instead of an arbitrary
threshold. `lse` lands at ~1e-7 relative error.

## Why it is slower at long S

The notebook's bandwidth table makes the ceiling explicit. Counting every K/V block
each program loads, at `S=8192` the kernel *requests* 4.36 GB in 2277 us — 1.92 TB/s,
or 123% of the A100's 1.555 TB/s HBM peak. That is only possible because most of
those requests are L2 hits, not DRAM traffic: the unavoidable DRAM footprint (Q, O,
K, V, LSE, once each) is just 84.9 MB. The kernel is riding cache reuse, and the
remaining gap to SDPA is compute-side pipelining, not a memory problem.

`l2_residency.csv` isolates that: holding FLOPs and grid shape fixed while growing
the KV footprint past L2's 40 MB shows throughput flattening rather than falling off
a cliff, since the causal schedule re-reads recent K/V blocks while they are still hot.

## Running it

Needs a CUDA GPU, Ampere or newer (SM 8.0+) — the SDPA FLASH baseline requires it.

```python
import torch
from fa2_fwd import fa2_fwd

q = torch.randn(1, 32, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1,  8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1,  8, 4096, 64, device="cuda", dtype=torch.float16)
o, lse = fa2_fwd(q, k, v, causal=True)
```

For the full picture run `fa_bench_notebook.ipynb` top to bottom; it needs
`torch`, `triton` and `pandas`. The sweep cell takes a few minutes.

## Status

Forward only. Planned next:

- **TTFT / TPOT benchmarks** — prefill vs decode latency, wiring the kernel into the
  KV-cache path in `model/`. The benchmarks here all measure prefill-shaped work.
- **Backward pass** — `fa2_bwd.py`, using the `lse` the forward already saves.

Not planned: Nsight Compute profiling. The dev box is a Lightning AI studio, where
`ncu` returns `ERR_NVGPUCTRPERM` because GPU performance counters are restricted, so
the memory analysis above is analytic plus `do_bench` timings instead of measured
counters.
