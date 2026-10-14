"""Export CPU dynamique int8 vérifié sur validation, pas choisi sur le test.
Poids source FP32 et checkpoints de reprise conservés dans Drive.
Utilise le backend quantifié PyTorch disponible ; API actuelle dépréciée.
"""
import json,pathlib,hashlib,time,statistics,copy,torch,warnings,platform
from dataclasses import asdict
from tokenizers import Tokenizer
from mandia_seq2seq import MandiaS2S,S2SConfig,generate
import mandia_stage4 as stage

@torch.inference_mode()
def replies(model,items,tok):
    output=[]
    for r in items:
        ids,_=generate(model,r['source'],1,2,24)
        output.append({'id':r['row']['id'],'prompt_fr':r['row']['prompt'],'reference_bci':r['row']['answer'],'generated_bci':tok.decode(ids),'eos':2 in ids})
    return output

@torch.inference_mode()
def cpu_benchmark(model,source):
    source=torch.tensor([source]);times=[]
    for repeat in range(4):
        memory,valid=model.encode(source);states=None;x=torch.tensor([[1]]);start=time.perf_counter()
        for offset in range(64):
            logits,states=model.decode(x,memory,valid,states,offset,True);x=logits[:,-1].argmax(-1,keepdim=True)
        if repeat:times.append(time.perf_counter()-start)
    median=statistics.median(times)
    return {'forced_decode_tokens':64,'threads':torch.get_num_threads(),'warmup':1,'repeats':3,'median_decode_seconds':median,'tokens_per_second':64/median,'device':'Colab CPU, not the user Mac','precision':'FP32 activations, dynamic int8 Linear weights if quantized'}

def main():
    root=stage.ROOT;folder=root/'export-cpu';folder.mkdir(exist_ok=True)
    report=json.loads((root/'rapport-etape4.json').read_text());candidate=report['candidate_by_validation']
    source=root/candidate/'model.pt';payload=torch.load(source,map_location='cpu',weights_only=True)
    tok=Tokenizer.from_file(str(root/'tokenizer.json'))
    assert stage.sha(root/'tokenizer.json')==payload['tokenizer_sha256']
    old_device=stage.DEVICE;stage.DEVICE=torch.device('cpu');threads=torch.get_num_threads();torch.set_num_threads(4)
    try:
        tok,groups=stage.prepare()
        model=MandiaS2S(S2SConfig(**payload['config'])).eval();model.load_state_dict(payload['state_dict'])
        f32valid=stage.evaluate(model,groups['validation'],torch.device('cpu'))
        f32reply=replies(model,groups['train']['reply'],tok)
        f32speed=cpu_benchmark(model,groups['test']['everyday'][0]['source'])
        engines=[x for x in ['x86','fbgemm','qnnpack'] if x in torch.backends.quantized.supported_engines]
        assert engines,'Aucun backend int8 disponible.'
        torch.backends.quantized.engine=engines[0]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            quant=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8,inplace=False).eval()
        qvalid=stage.evaluate(quant,groups['validation'],torch.device('cpu'))
        qreply=replies(quant,groups['train']['reply'],tok)
        qspeed=cpu_benchmark(quant,groups['test']['everyday'][0]['source'])
        delta=qvalid['selection_score']-f32valid['selection_score']
        unchanged=all(stage.canonical(a['generated_bci'])==stage.canonical(b['generated_bci']) for a,b in zip(f32reply,qreply))
        correct=all(stage.canonical(a['generated_bci'])==stage.canonical(a['reference_bci']) for a in qreply)
        accepted=delta<=.05 and unchanged and correct
        destination=folder/'model-dynamic-int8-cpu.pt'
        stage.atomic_torch(destination,{'config':payload['config'],'state_dict':quant.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256'],'format':'PyTorch eager dynamic int8 Linear state_dict; embedding/Conv1d/norm remain FP32','engine_tested':engines[0],'torch_version':str(torch.__version__),'source_model_sha256':stage.sha(source),'accepted_by_validation':accepted})
        # Relire le vrai export et reconstruire les couches quantifiées.
        saved=torch.load(destination,map_location='cpu',weights_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            restored=torch.ao.quantization.quantize_dynamic(MandiaS2S(S2SConfig(**saved['config'])).eval(),{torch.nn.Linear},dtype=torch.qint8)
        restored.load_state_dict(saved['state_dict'])
        assert replies(restored,groups['train']['reply'][:3],tok)==qreply[:3]
        result={'candidate':candidate,'accepted':accepted,'selection':'validation CE delta <=0.05 and all 14 trained reply outputs unchanged/correct; no test-based selection','validation_fp32':f32valid,'validation_int8':qvalid,'validation_delta':delta,'trained_replies_unchanged':unchanged,'trained_replies_correct':correct,'reply_count':len(qreply),'reply_outputs':qreply,'cpu_fp32':f32speed,'cpu_dynamic_int8':qspeed,'source_fp32_file_bytes':source.stat().st_size,'int8_file_bytes':destination.stat().st_size,'int8_file_sha256':stage.sha(destination),'size_fraction':destination.stat().st_size/source.stat().st_size,'engine':engines[0],'torch':torch.__version__,'safe_state_dict_reload_checked':True,'scope':'Experimental Colab CPU export; no guarantee of speed or backend support on other hardware; API deprecated, migration needed for future torch versions. Parameter count unchanged; serialization size reduced.'}
        stage.atomic_json(folder/'QUANTIFICATION-VERIFIEE.json',result)
        print('MANDIA_INT8_BEGIN',json.dumps(result,ensure_ascii=False),'MANDIA_INT8_END',flush=True)
        return result
    finally:
        stage.DEVICE=old_device;torch.set_num_threads(threads)
if __name__=='__main__':main()
