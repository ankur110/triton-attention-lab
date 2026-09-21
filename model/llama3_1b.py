import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import fa2_fwd, flash_decode

@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    dim: int = 2048             
    n_layers: int = 16
    n_heads: int = 32           
    n_kv_heads: int = 8         # key/value heads  -> GQA grouping = 32/8 = 4
    head_dim: int = 64
    intermediate_size: int = 8192
    norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 500000.0
    rope_factor: float = 32.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_old_context: int = 8192

    # cache sizing
    max_seq_len: int = 2048
    max_batch_size: int = 1
    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

# ----------------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root mean Square Norm
    """
    def __init__(self,dim: int,eps:float):
        super().__init__()

        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def forward(self,x):
        int_dtype=x.dtype
        x=x.float()
        var=x.pow(2).mean(-1,keepdim=True)
        x=x*torch.rsqrt(var+self.eps)
        return self.weight * x.to(int_dtype)


# ----------------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------------

def apply_llama3_scaling(inv_freq:torch.tensor,cfg:LlamaConfig)->torch.tensor:
    """
    scale for low,mid,high wavelength
    """
    low_wavelen=cfg.rope_old_context/cfg.rope_low_freq_factor #8192
    high_wavelen=cfg.rope_old_context/cfg.rope_high_freq_factor #2048

    # λj​=​2π/θj
    wavelen=2*math.pi/inv_freq 

    scaled= inv_freq/cfg.rope_factor

    #yarn method
    smooth=(cfg.rope_old_context/wavelen - cfg.rope_low_freq_factor)/(cfg.rope_high_freq_factor-cfg.rope_low_freq_factor)

    mid=(1-smooth)*scaled+smooth*inv_freq

    out=torch.where(wavelen>low_wavelen,scaled,inv_freq)
    is_mid=(wavelen>=high_wavelen)&(wavelen<=low_wavelen)
    return torch.where(is_mid,mid,out)

def precompute_rope(cfg:LlamaConfig,device):
    """
    precomputing the RoPE angles
    """
    idx=torch.arange(0,cfg.head_dim,2,device=device).float()
    inv_freq=1.0/(cfg.rope_theta**(idx/cfg.head_dim))
    inv_freq=apply_llama3_scaling(inv_freq,cfg)

    t=torch.arange(cfg.max_seq_len,device=device).float()
    freqs=torch.outer(t,inv_freq)
    emb=torch.cat((freqs,freqs),dim=-1)
    return emb.cos(),emb.sin()

def rotate_half(x):
    """
    half rotation
    """
    x1=x[...,:x.shape[-1]//2]
    x2=x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

def apply_rope(q,k,cos,sin):
    # cos/sin arrive as (T, head_dim); broadcast over (B, n_head, T, head_dim)
    cos=cos[None,None,:,:]
    sin=sin[None,None,:,:]
    return (q*cos)+(rotate_half(q)*sin),(k*cos)+(rotate_half(k)*sin)

# ----------------------------------------------------------------------------
# Attention (GQA + KV cache)
# ----------------------------------------------------------------------------

def repeat_kv(x,n_rep:int):
    """
    matching kv_heads to query heads by repeating
    """

    if n_rep==1:
        return x

    B,n_kv,T,hd=x.shape
    x=x[:,:,None,:,:].expand(B,n_kv,n_rep,T,hd)
    return x.reshape(B,n_kv*n_rep,T,hd)

class Attention(nn.Module):
    def __init__(self,cfg:LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)


        self.k_cache = None
        self.v_cache = None

        self.attn_impl = "sdpa" 
        

    def setup_cache(self, batch_size, max_seq_len, dtype, device):
        shape = (batch_size, self.n_kv_heads, max_seq_len, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    def cache_bytes(self):
        if self.k_cache is None:
            return 0
        return (self.k_cache.numel()+self.v_cache.numel())*self.k_cache.element_size()

    def forward(self, x, cos, sin, start_pos, use_cache):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        if use_cache:
            self.k_cache[:B, :, start_pos:start_pos + T] = k
            self.v_cache[:B, :, start_pos:start_pos + T] = v
            k = self.k_cache[:B, :, :start_pos + T]
            v = self.v_cache[:B, :, :start_pos + T]

        if self.attn_impl == "sdpa":            
            y = F.scaled_dot_product_attention(q, repeat_kv(k, self.n_rep),
                                               repeat_kv(v, self.n_rep), is_causal=T > 1)
        elif self.attn_impl == "sdpa_gqa":      
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, enable_gqa=True)
        elif self.attn_impl == "triton":
            if T > 1:                           
                assert start_pos == 0, "causal prefill assumes S_q == S_kv"
                y, _ = fa2_fwd(q, k, v, causal=True)
            else:                               
                y = flash_decode(q[:, :, 0], k, v, self.seq_lens[:B]).unsqueeze(2)
        else:
            raise ValueError(self.attn_impl)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

# ----------------------------------------------------------------------------
# MLP (SwiGLU)
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ----------------------------------------------------------------------------
# Block
# ----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x, cos, sin, start_pos, use_cache):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, start_pos, use_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------

class Llama(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.model = nn.ModuleDict(dict(
            embed_tokens=nn.Embedding(cfg.vocab_size, cfg.dim),
            layers=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]),
            norm=RMSNorm(cfg.dim, cfg.norm_eps),
        ))
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        self.lm_head.weight = self.model.embed_tokens.weight

        self.cos, self.sin = None, None

    def setup_caches(self, batch_size, max_seq_len, dtype, device):
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        for blk in self.model.layers:
            blk.self_attn.setup_cache(batch_size, max_seq_len, dtype, device)
            blk.self_attn.seq_lens = self.seq_lens          # one shared buffer

    def free_caches(self):
        for blk in self.model.layers:
            blk.self_attn.k_cache = None
            blk.self_attn.v_cache = None
            blk.self_attn.seq_lens = None

    def set_attn_impl(self, impl):
        for blk in self.model.layers:
            blk.self_attn.attn_impl = impl

    def cache_bytes(self):
        return sum(blk.self_attn.cache_bytes() for blk in self.model.layers)
        
    def init_rope(self, device):
        self.cos, self.sin = precompute_rope(self.cfg, device)

    def forward(self, idx, start_pos: int = 0, use_cache: bool = False, last_only: bool = False):
        B, T = idx.shape
        x = self.model.embed_tokens(idx)
        cos = self.cos[start_pos:start_pos + T].to(x.dtype)
        sin = self.sin[start_pos:start_pos + T].to(x.dtype)
        if use_cache:
            self.seq_lens.fill_(start_pos + T)   
        for blk in self.model.layers:
            x = blk(x, cos, sin, start_pos, use_cache)
        if last_only:
            x = x[:, -1:]
        x = self.model.norm(x)
        return self.lm_head(x)
    
    # -- weight loading -----------------------------------------------------
    
    @classmethod
    def from_hf(cls, repo="NousResearch/Llama-3.2-1B", device="cuda",
                dtype=torch.float16, max_seq_len=2048):
        from transformers import AutoModelForCausalLM

        cfg = LlamaConfig(max_seq_len=max_seq_len)
        model = cls(cfg)

        hf = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype)
        sd = hf.state_dict()
        if "lm_head.weight" not in sd:                       # tied checkpoints
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"]

        own = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in own}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        assert not missing, f"missing keys: {missing[:5]}"

        del hf, sd, filtered
        model.to(device=device, dtype=dtype).eval()
        model.init_rope(torch.device(device)) 
        return model, cfg


def count_params(model):
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:      # tied weights counted once
            continue
        seen.add(id(p))
        total += p.numel()
    return total