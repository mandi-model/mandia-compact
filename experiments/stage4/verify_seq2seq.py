import torch,tempfile,pathlib
from mandia_seq2seq import MandiaS2S,S2SConfig

def main():
    torch.set_num_threads(2)
    results=[]
    for variant in ['hybrid','attention']:
        torch.manual_seed(9)
        m=MandiaS2S(S2SConfig(vocab_size=32,dim=48,heads=6,kv_heads=2,ffn_dim=96,encoder_layers=2,decoder_layers=3,variant=variant,dropout=0)).eval()
        s=torch.tensor([[1,3,5,7],[1,4,6,0]]);t=torch.tensor([[1,9,10,11],[1,12,13,0]])
        full=m(s,t)
        future=t.clone();future[:,3]=20
        torch.testing.assert_close(full[:,:3],m(s,future)[:,:3],atol=1e-5,rtol=1e-5)
        extended=torch.cat([s,torch.zeros((2,3),dtype=torch.long)],1)
        torch.testing.assert_close(full,m(extended,t),atol=1e-5,rtol=1e-5)
        memory,valid=m.encode(s);states=None;values=[]
        for i in range(t.shape[1]):
            logits,states=m.decode(t[:,i:i+1],memory,valid,states,i,True);values.append(logits)
        torch.testing.assert_close(full,torch.cat(values,1),atol=1e-5,rtol=1e-5)
        first,states=m.decode(t[:,:2],memory,valid,cache=True)
        rest,_=m.decode(t[:,2:],memory,valid,states,2,True)
        torch.testing.assert_close(full,torch.cat([first,rest],1),atol=1e-5,rtol=1e-5)
        changed=s.clone();changed[:,1]=21
        assert (full-m(changed,t)).abs().max()>1e-5
        m.train();loss=m(s,t).square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.encoder[0].attn.q.weight.grad.abs().sum()>0
        results.append({'variant':variant,'causality':True,'padding_invariance':True,'token_and_chunk_cache':True,'source_effect':True,'encoder_gradients':True})
    print('S2S_VERIFICATION',results,flush=True)
    return results
if __name__=='__main__':main()
