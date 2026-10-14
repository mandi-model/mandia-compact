"""MandiaCompact S2S : implémentation Mandia, initialisation aléatoire.
Encodeur bidirectionnel et accès à la source dans chaque bloc du décodeur.
Hybride : 4 convolutions causales + 2 GQA ; référence : 6 GQA.
Les composants sont publiés ; cette combinaison n'est pas une nouveauté prouvée.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from mandia_model import RMSNorm, Rotary, GatedCausalConv, SwiGLU

@dataclass
class S2SConfig:
    vocab_size: int = 4096
    dim: int = 384
    heads: int = 6
    kv_heads: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 6
    ffn_dim: int = 768
    variant: str = 'hybrid'
    conv_kernel: int = 5
    context: int = 256
    dropout: float = .1

class Attention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.h,self.kvh,self.d=cfg.heads,cfg.kv_heads,cfg.dim//cfg.heads
        self.q=nn.Linear(cfg.dim,cfg.dim,bias=False)
        self.k=nn.Linear(cfg.dim,self.kvh*self.d,bias=False)
        self.v=nn.Linear(cfg.dim,self.kvh*self.d,bias=False)
        self.out=nn.Linear(cfg.dim,cfg.dim,bias=False)
        self.rope=Rotary(self.d)
    def keys(self,x):
        b,n,_=x.shape
        return tuple(layer(x).view(b,n,self.kvh,self.d).transpose(1,2) for layer in (self.k,self.v))
    def forward(self,x,source=None,valid=None,state=None,offset=0,cache=False,causal=False):
        b,n,_=x.shape
        q=self.q(x).view(b,n,self.h,self.d).transpose(1,2)
        past=0
        if source is not None:
            k,v=state if state is not None else self.keys(source)
        else:
            k,v=self.keys(x);q,k=self.rope(q,offset),self.rope(k,offset)
            if state is not None:
                past=state[0].shape[-2];k,v=torch.cat((state[0],k),-2),torch.cat((state[1],v),-2)
        mask=None
        if valid is not None:mask=valid[:,None,None,:]
        use_causal=causal and past==0
        if causal and past and n>1:
            mask=torch.arange(k.size(-2),device=x.device)[None,:] <= torch.arange(offset,offset+n,device=x.device)[:,None]
        y=F.scaled_dot_product_attention(q,k.repeat_interleave(self.h//self.kvh,1),v.repeat_interleave(self.h//self.kvh,1),attn_mask=mask,is_causal=use_causal)
        return self.out(y.transpose(1,2).contiguous().view(b,n,-1)),((k,v) if cache else None)

class EncoderBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.norm1=RMSNorm(cfg.dim);self.norm2=RMSNorm(cfg.dim)
        self.attn=Attention(cfg);self.ffn=SwiGLU(cfg);self.drop=nn.Dropout(cfg.dropout)
    def forward(self,x,valid):
        y,_=self.attn(self.norm1(x),valid=valid);x=x+self.drop(y)
        return x+self.drop(self.ffn(self.norm2(x)))

class DecoderBlock(nn.Module):
    def __init__(self,cfg,index):
        super().__init__();self.kind='conv' if cfg.variant=='hybrid' and index%3!=2 else 'gqa'
        self.norm1,self.norm2,self.norm3=RMSNorm(cfg.dim),RMSNorm(cfg.dim),RMSNorm(cfg.dim)
        self.mix=GatedCausalConv(cfg) if self.kind=='conv' else Attention(cfg)
        self.cross=Attention(cfg);self.ffn=SwiGLU(cfg);self.drop=nn.Dropout(cfg.dropout)
    def forward(self,x,source,valid,state=None,offset=0,cache=False):
        selfstate,crossstate=(None,None) if state is None else state
        if self.kind=='conv':y,selfstate=self.mix(self.norm1(x),selfstate,offset,cache)
        else:y,selfstate=self.mix(self.norm1(x),state=selfstate,offset=offset,cache=cache,causal=True)
        x=x+self.drop(y)
        y,crossstate=self.cross(self.norm2(x),source=source,valid=valid,state=crossstate,cache=cache)
        x=x+self.drop(y);x=x+self.drop(self.ffn(self.norm3(x)))
        return x,(selfstate,crossstate)

class MandiaS2S(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.embedding=nn.Embedding(cfg.vocab_size,cfg.dim)
        self.encoder=nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.encoder_layers)])
        self.decoder=nn.ModuleList([DecoderBlock(cfg,i) for i in range(cfg.decoder_layers)])
        self.encnorm,self.decnorm=RMSNorm(cfg.dim),RMSNorm(cfg.dim)
        self.apply(self._init)
        for name,p in self.named_parameters():
            if name.endswith(('out.weight','down.weight')):nn.init.normal_(p,std=.02/math.sqrt(2*(cfg.encoder_layers+cfg.decoder_layers)))
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding,nn.Conv1d)):nn.init.normal_(m.weight,std=.02)
    def encode(self,source,pad=0):
        valid=source!=pad
        if not valid.any(-1).all():raise ValueError('Source vide.')
        x=self.embedding(source)
        for block in self.encoder:x=block(x,valid)
        return self.encnorm(x),valid
    def decode(self,target,source,valid,states=None,offset=0,cache=False):
        if offset+target.shape[1]>self.cfg.context:raise ValueError('Contexte décodeur dépassé.')
        x=self.embedding(target);out=[]
        for i,block in enumerate(self.decoder):
            x,s=block(x,source,valid,None if states is None else states[i],offset,cache);out.append(s)
        logits=F.linear(self.decnorm(x),self.embedding.weight)
        return (logits,out) if cache else logits
    def forward(self,source,target):
        memory,valid=self.encode(source);return self.decode(target,memory,valid)
    def description(self):
        return {'config':asdict(self.cfg),'parameters':sum(p.numel() for p in self.parameters()),'decoder_mix':[b.kind for b in self.decoder],'initialization':'random, no pretrained model'}

@torch.inference_mode()
def generate(model,source,bos,eos,max_tokens=96):
    model.eval();dev=next(model.parameters()).device
    if isinstance(source,list):source=torch.tensor([source],device=dev)
    memory,valid=model.encode(source);states=None;token=torch.tensor([[bos]],device=dev);answer=[]
    for offset in range(max_tokens):
        logits,states=model.decode(token,memory,valid,states,offset,True)
        value=int(logits[0,-1].float().argmax());answer.append(value)
        if value==eos:break
        token=torch.tensor([[value]],device=dev)
    return answer,states

def state_bytes(states):
    if states is None:return 0
    if torch.is_tensor(states):return states.numel()*states.element_size()
    return sum(state_bytes(x) for x in states)
