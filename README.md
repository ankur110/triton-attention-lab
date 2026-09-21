# Flash Attention 2 in Triton

FlashAttention-2 forward and Flash-Decoding kernels written from scratch in Triton, with
grouped-query attention (GQA) and causal masking, benchmarked against PyTorch SDPA and a
naive eager implementation, then run end-to-end inside a from-scratch Llama-3.2-1B.

All numbers below are from an A100-SXM4-40GB (torch 2.8.0+cu128, triton 3.4.0),
fp16, `H=32` query heads, `H_kv=8` KV heads, `head_dim=64`, batch 1, causal.

## Layout

| Path | What it is |
| --- | --- |
| `kernels/fa2_fwd.py` | The prefill kernel. `_fa2_fwd` (the `@triton.jit` kernel) and `fa2_fwd` (the launcher). |
| `kernels/flash_decode.py` | The decode kernel. Split-KV Flash-Decoding for the `S_q == 1` case. |
| `benchmarks/fa_bench_notebook.ipynb` | Correctness checks, the block/warp/stage sweep, the prefill benchmark, and the bandwidth analysis. |
| `benchmarks/final_benchmark.ipynb` | End-to-end: decode bandwidth, split sweep, TTFT / TPOT, launch-overhead breakdown. |
| `model/llama3_1b.py` | Llama-3.2-1B from scratch (GQA, RoPE, KV cache). The consumer the kernels are aimed at. |
| `results/*.csv` | Committed results, so the numbers are readable without a GPU. Re-running a notebook overwrites them. |

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

### Prefill

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
noise of the hand-picked default. The sweep is in `results/sweep_configs.csv`.

Correctness is checked against a float64 reference computed a few heads at a time,
with the tolerance set to SDPA-flash's own error against that reference rather than a
fixed epsilon, so the kernel is held to "no worse than flash" instead of an arbitrary
threshold. `lse` lands at ~1e-7 relative error.

### Decode

`S_q = 1`, B=1, fp16, KV cache of `S` tokens. `p50` of `do_bench`:

| S | SDPA GQA | flash-decode | pure KV read |
| --- | --- | --- | --- |
| 4096 | 31.5 us | **22.6 us** | 32.8 us |
| 32768 | 90.8 us | **80.0 us** | 90.0 us |
| 131072 | 234.6 us | **234.0 us** | 267.1 us |

Takeaways:

- **Decode is bandwidth-bound, and the kernel hits the floor** — at every `S` it is at or
  under the time it takes to merely `.sum()` the same K/V tensors.
- **Splitting the KV axis is what buys it** — at `S=32k`, 1 split is 545 us, 8 splits 89 us,
  32 splits 81 us, 64 splits 85 us. 6.7x from parallelism alone, with a knee at 8.
- **The gain is a small-batch effect** — at B=16 splitting does nothing (846 us split vs
  847 us unsplit); 16x32 CTAs already fill the GPU, so there is no idle SM to hand work to.
- **The prefill kernel is the wrong tool for decode** — driven at `S_q = 1` it flatlines at
  ~70 GB/s regardless of `S`, 15x slower than flash-decode at 128k. Hence a separate kernel.
- **TPOT is flat across context** — 13.4 ms at both 1k and 128k context, while plain SDPA
  degrades 10.8 -> 53.8 ms. That flatness is the whole point of flash-decode.
- **Those TPOT numbers are CPU-bound, not kernel-bound** — at 8k context the triton path is
  the *fastest* on GPU time (5.00 ms vs SDPA's 7.39 ms) and the *slowest* on wall clock,
  because its launch gap is 11.2 ms. A 32% GPU-time win that the eager Python loop spends.
  CUDA graphs would recover it.
- **TTFT is a wash** — prefill at 8192 tokens is 138 ms vs SDPA's 129 ms, matching the ~25%
  prefill-kernel gap above once it is diluted by the rest of the model.
- **End-to-end correctness holds** — 64/64 greedy tokens match the SDPA reference, max logit
  diff 1.6e-2 (fp16 accumulation noise).

## Why it is slower at long S

The notebook's bandwidth table makes the ceiling explicit. Counting every K/V block
each program loads, at `S=8192` the kernel *requests* 4.36 GB in 2277 us — 1.92 TB/s,
or 123% of the A100's 1.555 TB/s HBM peak. That is only possible because most of
those requests are L2 hits, not DRAM traffic: the unavoidable DRAM footprint (Q, O,
K, V, LSE, once each) is just 84.9 MB. The kernel is riding cache reuse, and the
remaining gap to SDPA is compute-side pipelining, not a memory problem.

`results/l2_residency.csv` isolates that: holding FLOPs and grid shape fixed while growing
the KV footprint past L2's 40 MB shows throughput flattening rather than falling off
a cliff, since the causal schedule re-reads recent K/V blocks while they are still hot.

## Running it

Needs a CUDA GPU, Ampere or newer (SM 8.0+) — the SDPA FLASH baseline requires it.

```python
import torch
from kernels import fa2_fwd

q = torch.randn(1, 32, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1,  8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1,  8, 4096, 64, device="cuda", dtype=torch.float16)
o, lse = fa2_fwd(q, k, v, causal=True)
```

Run the notebooks in `benchmarks/` from inside that directory; they need `torch`,
`triton`, `pandas` and (for `final_benchmark.ipynb`) `transformers`. The sweep cell
takes a few minutes.

## Status

Forward only: prefill and decode, both wired into `model/`. Planned next:

- **Backward pass** — `kernels/fa2_bwd.py`, using the `lse` the forward already saves.

Not planned: Nsight Compute profiling. The dev box is a Lightning AI studio, where
`ncu` returns `ERR_NVGPUCTRPERM` because GPU performance counters are restricted, so
the memory analysis above is analytic plus `do_bench` timings instead of measured
counters.
