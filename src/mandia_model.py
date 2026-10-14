"""MandiaCompact v0 : prototype causal écrit pour Mandia, poids initialisés aléatoirement.
Briques connues : RMSNorm, RoPE, SwiGLU, GQA et convolution causale courte.
La combinaison proposée est une hypothèse expérimentale, pas une nouveauté démontrée.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 384
    layers: int = 8
    heads: int = 6
    kv_heads: int = 2
    ffn_dim: int = 768
    context: int = 512
    conv_kernel: int = 5
    attention_every: int = 4
    variant: str = "hybrid"
    tie_embeddings: bool = True
    def __post_init__(self):
        assert self.dim % self.heads == 0
        assert self.heads % self.kv_heads == 0
        assert (self.dim // self.heads) % 2 == 0
        assert self.variant in {"hybrid", "attention", "conv_only"}
        assert self.conv_kernel >= 1 and self.attention_every >= 1

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x.float() * scale).to(x.dtype) * self.weight.to(x.dtype)

class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float()/dim)), persistent=False)
    def forward(self, x, offset):
        positions = torch.arange(offset, offset+x.size(-2), device=x.device, dtype=torch.float32)
        angle = positions[:, None] * self.inv_freq[None, :]
        cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2)

class GroupedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg.heads, cfg.kv_heads, cfg.dim//cfg.heads
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.kv_heads*self.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.kv_heads*self.head_dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.rope = Rotary(self.head_dim)
    def forward(self, x, state=None, offset=0, cache=False):
        batch, length, _ = x.shape
        q = self.q(x).view(batch,length,self.heads,self.head_dim).transpose(1,2)
        k = self.k(x).view(batch,length,self.kv_heads,self.head_dim).transpose(1,2)
        v = self.v(x).view(batch,length,self.kv_heads,self.head_dim).transpose(1,2)
        q, k = self.rope(q, offset), self.rope(k, offset)
        past = 0
        if state is not None:
            past = state[0].size(-2)
            k, v = torch.cat((state[0],k),dim=-2), torch.cat((state[1],v),dim=-2)
        new_state = (k, v) if cache else None
        # Le cache reste en kv_heads ; cette implémentation portable répète ensuite K/V pour SDPA.
        kr = k.repeat_interleave(self.heads//self.kv_heads,dim=1)
        vr = v.repeat_interleave(self.heads//self.kv_heads,dim=1)
        mask, causal = None, past == 0
        if past and length > 1:
            qi = torch.arange(length,device=x.device)[:,None] + past
            ki = torch.arange(k.size(-2),device=x.device)[None,:]
            mask = ki <= qi
        y = F.scaled_dot_product_attention(q,kr,vr,attn_mask=mask,dropout_p=0.0,is_causal=causal)
        y = y.transpose(1,2).contiguous().view(batch,length,-1)
        return self.out(y), new_state

class GatedCausalConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kernel = cfg.conv_kernel
        self.proj = nn.Linear(cfg.dim,cfg.dim,bias=False)
        self.conv = nn.Conv1d(cfg.dim,cfg.dim,cfg.conv_kernel,groups=cfg.dim,bias=False)
        self.gate_weight = nn.Parameter(torch.zeros(cfg.dim))
        self.gate_bias = nn.Parameter(torch.zeros(cfg.dim))
        self.out = nn.Linear(cfg.dim,cfg.dim,bias=False)
    def forward(self, x, state=None, offset=0, cache=False):
        u = self.proj(x).transpose(1,2)
        if state is None:
            state = u.new_zeros(u.size(0),u.size(1),self.kernel-1)
        sequence = torch.cat((state,u),dim=-1)
        y = self.conv(sequence).transpose(1,2)
        gate = torch.sigmoid(x*self.gate_weight.to(x.dtype)+self.gate_bias.to(x.dtype))
        y = F.silu(y)*gate
        new_state = sequence[:,:,-(self.kernel-1):] if self.kernel > 1 else sequence[:,:,:0]
        return self.out(y), new_state.clone() if cache else None

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.dim,cfg.ffn_dim,bias=False)
        self.up = nn.Linear(cfg.dim,cfg.ffn_dim,bias=False)
        self.down = nn.Linear(cfg.ffn_dim,cfg.dim,bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x))*self.up(x))

class Block(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        attention = cfg.variant == "attention" or (cfg.variant == "hybrid" and (index+1)%cfg.attention_every == 0)
        self.kind = "gqa" if attention else "conv"
        self.norm1, self.norm2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.mix = GroupedAttention(cfg) if attention else GatedCausalConv(cfg)
        self.ffn = SwiGLU(cfg)
    def forward(self, x, state=None, offset=0, cache=False):
        delta, state = self.mix(self.norm1(x),state,offset,cache)
        x = x+delta
        x = x+self.ffn(self.norm2(x))
        return x, state

class MandiaLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size,cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg,i) for i in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.dim,cfg.vocab_size,bias=False)
        self.apply(self._init)
        for name, parameter in self.named_parameters():
            if name.endswith(("mix.out.weight", "ffn.down.weight")):
                nn.init.normal_(parameter,std=0.02/math.sqrt(2*cfg.layers))
    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Linear,nn.Embedding,nn.Conv1d)):
            nn.init.normal_(module.weight,std=0.02)
    def forward(self, input_ids, states=None, offset=0, cache=False):
        if offset+input_ids.size(1) > self.cfg.context:
            raise ValueError("Contexte dépassé : raccourcir la demande ou la génération.")
        x, new_states = self.embedding(input_ids), []
        for i, block in enumerate(self.blocks):
            x, state = block(x,None if states is None else states[i],offset,cache)
            new_states.append(state)
        x = self.norm(x)
        logits = F.linear(x,self.embedding.weight) if self.head is None else self.head(x)
        return (logits,new_states) if cache else logits
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
    def description(self):
        return {"config":asdict(self.cfg),"parameters":self.parameter_count(),
                "blocks":[b.kind for b in self.blocks],"initialization":"random, no pretrained weights"}

@torch.inference_mode()
def generate_ids(model, prefix, eos_id, max_new_tokens=64, temperature=0.0):
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor([prefix],dtype=torch.long,device=device)
    logits, states = model(x,cache=True)
    generated, offset = [], len(prefix)
    for i in range(min(max_new_tokens,model.cfg.context-len(prefix))):
        last = logits[0,-1].float()
        token = int(last.argmax()) if temperature <= 0 else int(torch.multinomial(F.softmax(last/temperature,dim=-1),1))
        generated.append(token)
        if token == eos_id:
            break
        if i+1 < min(max_new_tokens,model.cfg.context-len(prefix)):
            logits, states = model(torch.tensor([[token]],device=device),states,offset,cache=True)
            offset += 1
    return generated

def cache_bytes(states):
    return sum(t.numel()*t.element_size() for state in states for t in (state if isinstance(state,tuple) else [state]) if t is not None)
