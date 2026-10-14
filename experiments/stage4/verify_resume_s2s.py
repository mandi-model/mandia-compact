"""Test de la vraie boucle : interruption après sauvegarde puis reprise 100+4."""
import pathlib,tempfile,torch,random,json
import mandia_stage4 as stage
from mandia_seq2seq import MandiaS2S,S2SConfig
def main():
    torch.set_num_threads(2);stage.DEVICE=torch.device('cpu')
    cfg=S2SConfig(vocab_size=32,dim=48,heads=6,kv_heads=2,ffn_dim=96,encoder_layers=1,decoder_layers=2,dropout=.1)
    items=[{'source':[5,7+i,8+i,2],'target':[7+i,8+i,2],'row':{'id':str(i)}} for i in range(6)]
    groups={'train':{'monolingual':items},'validation':{d:items for d in ['everyday','software','gospel','lexical']}}
    def model():torch.manual_seed(13);random.seed(13);return MandiaS2S(cfg)
    with tempfile.TemporaryDirectory() as d:
        a,b=pathlib.Path(d)/'full',pathlib.Path(d)/'resume'
        stage.train(model(),groups,a,'resume-test',torch.device('cpu'),steps=104)
        original=stage.atomic_torch
        class Interrupted(Exception):pass
        def stop(path,payload):
            original(path,payload)
            if pathlib.Path(path).name=='checkpoint-last.pt' and payload.get('step')==100:raise Interrupted()
        stage.atomic_torch=stop
        try:stage.train(model(),groups,b,'resume-test',torch.device('cpu'),steps=104)
        except Interrupted:pass
        finally:stage.atomic_torch=original
        stage.train(model(),groups,b,'resume-test',torch.device('cpu'),steps=104)
        x=torch.load(a/'checkpoint-last.pt',weights_only=True);y=torch.load(b/'checkpoint-last.pt',weights_only=True)
        assert all(torch.equal(x['model'][k],y['model'][k]) for k in x['model'])
        assert torch.equal(x['sampler_rng'],y['sampler_rng']) and torch.equal(x['torch_rng'],y['torch_rng'])
        assert x['real_tokens']==y['real_tokens'] and x['supervised_tokens']==y['supervised_tokens']
        for k,v in x['optimizer']['state'].items():
            assert all(torch.equal(t,y['optimizer']['state'][k][n]) if torch.is_tensor(t) else t==y['optimizer']['state'][k][n] for n,t in v.items())
    result={'cpu_actual_training_resume':'104 uninterrupted == 100+4 resumed, bitwise equal weights and AdamW','sampler_and_dropout_rng':'equal','token_counters':'equal','gpu_scope':'CUDA RNG saved; GPU bitwise equivalence not asserted'}
    print('S2S_RESUME_TEST',json.dumps(result),flush=True)
    return result
if __name__=='__main__':main()
