import torch
import triton
import triton.language as tl


@triton.jit
def _fa2_fwd(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    S_q, S_kv,
    sm_scale,
    H: tl.constexpr,
    N_REP: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    General FA-2 forward. LSE is assumed contiguous (B, H, S_q) fp32.
    Invariant relied on for causal: the first key block a program processes
    is [0, BLOCK_N), which every query row can see at least partially.
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H
    kv_h = h // N_REP

    Q += b * stride_qb + h    * stride_qh
    K += b * stride_kb + kv_h * stride_kh
    V += b * stride_vb + kv_h * stride_vh
    O += b * stride_ob + h    * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m[:, None] < S_q

    q_ptr = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptr, mask=mask_m, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    lo_diag = pid_m * BLOCK_M
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, S_kv)
    else:
        hi = S_kv

    for start_n in range(0, lo_diag if CAUSAL else S_kv, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_kv = offs_n[:, None] < S_kv
        k_ptr = K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptr = V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptr, mask=mask_kv, other=0.0)
        v = tl.load(v_ptr, mask=mask_kv, other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        s = tl.where(offs_n[None, :] < S_kv, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for start_n in range(lo_diag if CAUSAL else 0, hi if CAUSAL else 0, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_kv = offs_n[:, None] < S_kv
        k_ptr = K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptr = V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptr, mask=mask_kv, other=0.0)
        v = tl.load(v_ptr, mask=mask_kv, other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        causal_ok = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < S_kv)
        s = tl.where(causal_ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    o_ptr = O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptr, o.to(q.dtype), mask=mask_m)
    tl.store(LSE + pid_bh * S_q + offs_m, lse, mask=offs_m < S_q)


def fa2_fwd(q, k, v, causal=False, sm_scale=None,
            BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2):
    """
    q: (B, H, S_q, d)   k, v: (B, H_kv, S_kv, d)
    returns o: (B, H, S_q, d) same dtype as q,  lse: (B, H, S_q) fp32
    """
    B, H, S_q, d = q.shape
    _, H_kv, S_kv, _ = k.shape

    assert k.shape == v.shape, "K and V must have identical shapes"
    assert k.shape[0] == B, "Batch dimensions must match"
    assert H % H_kv == 0, f"Query heads ({H}) must be a multiple of KV heads ({H_kv})"
    assert (d & (d - 1)) == 0 and d >= 16, "Head dimension must be a power of 2 and >= 16"
    assert q.dtype == k.dtype == v.dtype, "Tensors must share dtype"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA"

    if causal:
        assert S_q == S_kv, "Causal implementation currently assumes identical sequence lengths"

    if sm_scale is None:
        sm_scale = 1.0 / (d ** 0.5)

    o = torch.empty_like(q)
    lse = torch.empty((B, H, S_q), dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(S_q, BLOCK_M), B * H)
    N_REP = H // H_kv

    _fa2_fwd[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        S_q, S_kv,
        sm_scale,
        H=H,
        N_REP=N_REP,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o, lse
