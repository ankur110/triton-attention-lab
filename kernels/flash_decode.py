import math
import torch
import triton
import triton.language as tl


@triton.jit
def _split_kv_kernel(
    Q, K, V, SeqLens, O_part, LSE_part,
    stride_qb, stride_qh,                  # q [B, Hq, D], last stride = 1
    stride_kb, stride_kh, stride_ks,       # k [B, Hkv, S_max, D]
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,       # O_part [B, Hq, NUM_SPLITS, D]
    stride_lb, stride_lh,                  # LSE_part [B, Hq, NUM_SPLITS]
    sm_scale,
    G: tl.constexpr,           # query heads per kv head (4)
    BLOCK_H: tl.constexpr,     # G padded to 16 for tl.dot
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b     = tl.program_id(0)
    kv_h  = tl.program_id(1)
    split = tl.program_id(2)

    seq_len = tl.load(SeqLens + b)

    # TODO 1: this split's token range
    #   chunk = cdiv(cdiv(seq_len, NUM_SPLITS), BLOCK_N) * BLOCK_N   (aligned to BLOCK_N)
    #   start = split * chunk ;  end = min(start + chunk, seq_len)
    chunk = tl.cdiv(tl.cdiv(seq_len,NUM_SPLITS),BLOCK_N)*BLOCK_N
    start = split*chunk
    end   = tl.minimum(start+chunk,seq_len)

    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    h_mask  = offs_h < G                   # rows 0..G-1 real, rest padding
    q_heads = kv_h * G + offs_h            # GQA packing: this group's query heads

    # TODO 2: load packed Q block [BLOCK_H, D]
    #   address: Q + b*stride_qb + q_heads[:, None]*stride_qh + offs_d[None, :]
    #   mask with h_mask[:, None], other=0.0
    K+=b*stride_kb + kv_h*stride_kh 
    V+=b*stride_vb + kv_h*stride_vh 
    q_ptrs = Q+b*stride_qb+q_heads[:,None]*stride_qh+offs_d[None,:]
    q=tl.load(q_ptrs,mask=h_mask[:, None],other=0.0)

    m_i = tl.full([BLOCK_H], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)

    for n0 in range(start, end, BLOCK_N):
        kv_pos  = n0 + offs_n
        kv_mask = kv_pos < end
        # TODO 3: load K and V tiles [BLOCK_N, D] for (b, kv_h) -- ONE load serves all G rows
        #   address: K + b*stride_kb + kv_h*stride_kh + kv_pos[:, None]*stride_ks + offs_d[None, :]
        k_ptrs=K+kv_pos[:, None]*stride_ks + offs_d[None, :]
        v_ptrs=V+kv_pos[:, None]*stride_vs + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)


        # TODO 4: s = tl.dot(q, tl.trans(k)) * sm_scale          -> [BLOCK_H, BLOCK_N]
        #         s = tl.where(kv_mask[None, :], s, float('-inf'))  (scores, not the load!)
        s=tl.dot(q,tl.trans(k))*sm_scale
        s=tl.where(kv_mask[None,:],s,float('-inf'))
        # TODO 5: online softmax per row (same as L3):
        #         m_new, scale = exp(m_i - m_new), p = exp(s - m_new[:, None]),
        #         l_i = l_i*scale + sum(p), acc = acc*scale[:, None] + dot(p.to(fp16), v)
        m_new=tl.maximum(m_i,tl.max(s,axis=1))
        scale=tl.exp(m_i-m_new)
        p=tl.exp(s-m_new[:,None])
        l_i=l_i*scale + tl.sum(p,axis=1)
        acc=acc*scale[:,None]+tl.dot(p.to(tl.float16),v)
        m_i=m_new

    # TODO 6: finalize with the empty-split guard
    #   l_i > 0 : o = acc / l_i[:, None],  lse = m_i + log(l_i)
    #   l_i == 0: o = 0,                   lse = -inf
    valid = l_i > 0.0
    o = tl.where(valid[:, None], acc / l_i[:, None], 0.0)
    lse = tl.where(valid, m_i + tl.math.log(l_i), float('-inf'))
    # TODO 7: store o and lse for real rows only (mask h_mask)
    #   O_part  + b*stride_ob + q_heads[:, None]*stride_oh + split*stride_os + offs_d[None, :]
    #   LSE_part + b*stride_lb + q_heads*stride_lh + split
    o_ptrs = O_part + b * stride_ob + q_heads[:, None] * stride_oh + split * stride_os + offs_d[None, :]
    lse_ptrs = LSE_part + b * stride_lb + q_heads * stride_lh + split
    
    tl.store(o_ptrs, o, mask=h_mask[:, None])
    tl.store(lse_ptrs, lse, mask=h_mask)


@triton.jit
def _combine_kernel(
    O_part, LSE_part, Out,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh,
    stride_outb, stride_outh,              # Out [B, Hq, D]
    D: tl.constexpr, NUM_SPLITS: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)         # BLOCK_S = next_pow2(NUM_SPLITS)
    offs_d = tl.arange(0, D)
    s_mask = offs_s < NUM_SPLITS

    # TODO 8: load lse [BLOCK_S]; padding -> -inf
    lse_ptrs=LSE_part+b*stride_lb+h*stride_lh+offs_s
    lse = tl.load(lse_ptrs, mask=s_mask, other=float('-inf'))
    # TODO 9: lse_max = max(lse); w = exp(lse - lse_max); w = w / sum(w)
    lse_max=tl.max(lse,axis=0)
    w=tl.exp(lse-lse_max)
    w=w/tl.sum(w,axis=0)
    # TODO 10: load partial O [BLOCK_S, D] (padding -> 0),
    #          out = sum(w[:, None] * o, axis=0); store as fp16 to Out[b, h, :]

    o_ptrs = O_part + b * stride_ob + h * stride_oh + offs_s[:, None] * stride_os + offs_d[None, :]
    o = tl.load(o_ptrs, mask=s_mask[:, None], other=0.0)
    out = tl.sum(w[:, None] * o, axis=0)
    
    out_ptrs = Out + b * stride_outb + h * stride_outh + offs_d
    tl.store(out_ptrs, out.to(Out.dtype.element_ty))


def flash_decode(q, k, v, seq_lens, num_splits=None, BLOCK_N=64):
    """q [B, Hq, D], k/v [B, Hkv, S_max, D] fp16, seq_lens [B] int32 -> out [B, Hq, D]"""
    B, Hq, D = q.shape
    Hkv, S_max = k.shape[1], k.shape[2]
    G = Hq // Hkv
    if num_splits is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = max(1, min(triton.cdiv(n_sm, B * Hkv), triton.cdiv(S_max, BLOCK_N)))

    O_part   = torch.empty(B, Hq, num_splits, D, dtype=torch.float32, device=q.device)
    LSE_part = torch.empty(B, Hq, num_splits,    dtype=torch.float32, device=q.device)
    out      = torch.empty(B, Hq, D, dtype=q.dtype, device=q.device)

    _split_kv_kernel[(B, Hkv, num_splits)](
        q, k, v, seq_lens, O_part, LSE_part,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        O_part.stride(0), O_part.stride(1), O_part.stride(2),
        LSE_part.stride(0), LSE_part.stride(1),
        1.0 / math.sqrt(D),
        G=G, BLOCK_H=max(16, triton.next_power_of_2(G)), BLOCK_N=BLOCK_N, D=D,
        NUM_SPLITS=num_splits,
    )
    _combine_kernel[(B, Hq)](
        O_part, LSE_part, out,
        O_part.stride(0), O_part.stride(1), O_part.stride(2),
        LSE_part.stride(0), LSE_part.stride(1),
        out.stride(0), out.stride(1),
        D=D, NUM_SPLITS=num_splits, BLOCK_S=triton.next_power_of_2(num_splits),
    )
    return out