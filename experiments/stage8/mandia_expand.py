"""Function-preserving FFN expansion of Mandia's own SwiGLU model.

Inspired by Net2Net (Chen et al., ICLR 2016), but this is zero-output channel
extension, not their neuron replication algorithm and not a novelty claim.
No external model weights; no tokenizer or attention/cache layout changes.
"""
from dataclasses import asdict
import torch
from mandia_seq2seq import MandiaS2S, S2SConfig

def widen_ffn(parent, new_ffn_dim=1536, seed=43):
    cfg=asdict(parent.cfg)
    old=cfg['ffn_dim']
    if new_ffn_dim<=old:
        raise ValueError('New FFN dimension must exceed the original')
    cfg['ffn_dim']=new_ffn_dim
    # Preserve the caller's RNG; fair arms reset training RNG independently.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        widened=MandiaS2S(S2SConfig(**cfg))
    state=widened.state_dict()
    for name,tensor in parent.state_dict().items():
        original=tensor.detach().cpu()
        if '.ffn.gate.weight' in name or '.ffn.up.weight' in name:
            state[name][:old].copy_(original)
        elif '.ffn.down.weight' in name:
            state[name].zero_()
            state[name][:,:old].copy_(original)
        else:
            if state[name].shape!=original.shape:
                raise ValueError('Unexpected changed shape: '+name)
            state[name].copy_(original)
    widened.load_state_dict(state,strict=True)
    return widened

def verify():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    c=S2SConfig(vocab_size=48,dim=48,heads=6,kv_heads=2,
                encoder_layers=1,decoder_layers=3,ffn_dim=64,dropout=.1)
    a=MandiaS2S(c).eval();b=widen_ffn(a,128).eval()
    x=torch.tensor([[4,8,12,2,0],[4,11,13,15,2]])
    y=torch.tensor([[1,17,18],[1,20,21]])
    with torch.inference_mode():
        za=a(x,y);zb=b(x,y)
        delta=float((za-zb).abs().max())
        torch.testing.assert_close(za,zb,atol=2e-6,rtol=2e-5)
        assert torch.equal(za.argmax(-1),zb.argmax(-1))
        enc,valid=b.encode(x)
        states=None;pieces=[]
        for i in range(y.shape[1]):
            z,states=b.decode(y[:,i:i+1],enc,valid,states,i,True)
            pieces.append(z)
        cached=torch.cat(pieces,dim=1)
        torch.testing.assert_close(cached,zb,atol=2e-6,rtol=2e-5)
    b.train();loss=torch.nn.functional.cross_entropy(b(x,y).reshape(-1,48),torch.tensor([17,18,2,20,21,2]));loss.backward()
    extra_grads={n:float(p.grad[:,64:].abs().sum()) for n,p in b.named_parameters() if '.ffn.down.weight' in n}
    assert all(v>0 for v in extra_grads.values()),extra_grads
    return {'fixture':'random tiny CPU model, not trained weights or quality benchmark',
            'preserves_eval_logits_with_tolerance':True,'max_absolute_error':delta,
            'preserves_argmax':True,'cached_decode_matches_full':True,
            'all_added_down_columns_receive_nonzero_gradient':True,
            'extra_down_gradient_l1':extra_grads,'gpu_bf16_equivalence_not_yet_verified':True}

if __name__=='__main__':
    import json,pathlib,gc
    result=verify();counts={}
    for size in [768,1536,2304]:
        m=MandiaS2S(S2SConfig(ffn_dim=size));counts[str(size)]=sum(p.numel() for p in m.parameters());del m;gc.collect()
    result['parameter_counts_by_ffn_dimension']=counts
    path=pathlib.Path(__file__).with_name('VERIFICATION-ARCHITECTURE-CPU.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
