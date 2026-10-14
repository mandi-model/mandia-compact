"""Expérience S2S contrôlée : nouveaux poids, mêmes données/budgets pour 2 architectures.
Toutes les sauvegardes (poids, AdamW, RNG, compteur) vont directement dans Drive.
"""
import json,pathlib,hashlib,random,time,math,gc,unicodedata,re,collections,contextlib
from dataclasses import asdict
import torch
from torch.nn import functional as F
from mandia_seq2seq import S2SConfig,MandiaS2S,generate,state_bytes
ROOT=pathlib.Path('/content/drive/MyDrive/Mandia/Sauvegarde-langues-MandiaCompact-2026-10-13/colab-runtime/mandia-compact-s2s-etape4')
STEPS,WARMUP,BATCH,LR=6000,200,16,3e-4
SPECIAL=['<pad>','<bos>','<eos>','<translate>','<reply>','<denoise>','<mask>']
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def canonical(s):return ' '.join(unicodedata.normalize('NFC',s).casefold().replace('’',"'").split())
def atomic_json(p,data):
    p=pathlib.Path(p);tmp=p.with_suffix('.json.part');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
def atomic_torch(p,data):
    p=pathlib.Path(p);tmp=p.with_suffix('.pt.part');torch.save(data,tmp);tmp.replace(p)
def context():return torch.autocast('cuda',dtype=torch.bfloat16) if DEVICE.type=='cuda' else contextlib.nullcontext()
def sync():
    if DEVICE.type=='cuda':torch.cuda.synchronize()

def prepare():
    from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
    rows=[json.loads(l) for l in (ROOT/'parallel-prepare.jsonl').read_text().splitlines()]
    replies=[json.loads(l) for l in (ROOT/'replies-prepare.jsonl').read_text().splitlines()]
    protected={canonical(r[k]) for r in rows if r['split']!='train' for k in ['prompt','answer']}
    for split in ['train','validation','test']:
        for other in ['train','validation','test']:
            if split>=other:continue
            for field in ['prompt','answer']:
                a={canonical(r[field]) for r in rows if r['split']==split}
                b={canonical(r[field]) for r in rows if r['split']==other}
                assert not a&b,(split,other,field)
    assert all(not any(canonical(r[k]) in protected for k in ['prompt','answer']) for r in replies)
    mono=[];waxal=ROOT/'data/waxal-baoule-text.jsonl'
    if waxal.exists():
        for line in waxal.read_text().splitlines():
            r=json.loads(line)
            for j,paragraph in enumerate(re.split(r'\n+|(?<=[.!?])\s+',r['text'])):
                t=' '.join(paragraph.split())
                if 12<=len(t)<=900 and canonical(t) not in protected:
                    mono.append({'id':r['id']+f'-{j}','prompt':t,'answer':t,'task':'denoise','domain':'monolingual','split':'train','source_document':r['id']})
    assert mono,'Extraire les transcriptions WAXAL avant cette expérience.'
    # Exclusion des longues références holdout apparaissant à l'intérieur d'une phrase.
    hold=[p for p in protected if len(p)>=25]
    mono=[r for r in mono if not any(p in canonical(r['answer']) for p in hold)]
    tokenpath=ROOT/'tokenizer.json'
    if tokenpath.exists():tok=Tokenizer.from_file(str(tokenpath))
    else:
        tok=Tokenizer(models.BPE());tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
        tr=trainers.BpeTrainer(vocab_size=4096,min_frequency=2,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=SPECIAL)
        texts=(r[k] for r in rows+replies+mono if r['split']=='train' for k in ['prompt','answer'])
        tok.train_from_iterator(texts,trainer=tr);tok.save(str(tokenpath))
    assert tok.decode(tok.encode('ɛ́ ɔ̀ ɲ á à ’ Français — Baoulé').ids)=='ɛ́ ɔ̀ ɲ á à ’ Français — Baoulé'
    groups={s:collections.defaultdict(list) for s in ['train','validation','test']};encoded=[];excluded=collections.Counter()
    for r in rows+replies+mono:
        src=[tok.token_to_id('<'+r['task']+'>')]+tok.encode(r['prompt']).ids+[2]
        target=tok.encode(r['answer']).ids+[2]
        if len(src)>160 or len(target)>192:
            excluded[r['domain']+'/length']+=1;continue
        item={'source':src,'target':target,'row':r};groups[r['split']][r['domain']].append(item)
        encoded.append({'id':r['id'],'split':r['split'],'domain':r['domain'],'source_tokens':len(src),'target_tokens':len(target)})
    for s in ['train','validation','test']:
        for d in ['everyday','software','gospel','lexical']:assert groups[s][d],(s,d)
    assert groups['train']['monolingual'] and groups['train']['reply']
    audit={'rows':{s:{d:len(items) for d,items in ds.items()} for s,ds in groups.items()},'excluded':dict(excluded),'tokenizer_sha256':sha(tokenpath),'monolingual_source':'Google WAXAL official training split, text columns only','no_holdout_tokenizer_training':True,'source_max':160,'target_max':192,'identity_denoising_is_not_translation':True}
    atomic_json(ROOT/'DATA-AUDIT.json',audit)
    (ROOT/'split-tokenise.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in encoded))
    print('DATA_ETAPE4',json.dumps(audit),flush=True)
    return tok,groups

def batch(items,device,generator=None,denoise=False):
    ss=[list(r['source']) for r in items];tt=[r['target'] for r in items]
    if denoise:
        for s in ss:
            for i in range(1,len(s)-1):
                if float(torch.rand((),generator=generator))<.25:s[i]=6
    source=torch.zeros((len(ss),max(map(len,ss))),dtype=torch.long,device=device)
    target=torch.zeros((len(tt),max(map(len,tt))),dtype=torch.long,device=device)
    labels=torch.full_like(target,-100)
    for i,(s,t) in enumerate(zip(ss,tt)):
        source[i,:len(s)]=torch.tensor(s,device=device);target[i,:len(t)]=torch.tensor([1]+t[:-1],device=device);labels[i,:len(t)]=torch.tensor(t,device=device)
    return source,target,labels,sum(map(len,ss))+sum(map(len,tt))
def choose(items,n,g):return [items[int(i)] for i in torch.randint(len(items),(n,),generator=g)]
def loss(model,source,target,labels,smoothing=0):
    logits=model(source,target)
    return F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100,reduction='sum',label_smoothing=smoothing),int((labels!=-100).sum())

@torch.inference_mode()
def evaluate(model,groups,device):
    model.eval();metrics={}
    for domain in ['everyday','software','gospel','lexical']:
        total,count=0.,0
        for start in range(0,len(groups[domain]),16):
            s,t,y,_=batch(groups[domain][start:start+16],device)
            with context():v,n=loss(model,s,t,y)
            total+=float(v);count+=n
        metrics[domain]={'loss':total/count,'supervised_tokens':count,'rows':len(groups[domain])}
    return {'selection_score':sum(x['loss'] for x in metrics.values())/4,'domains':metrics}

def train(model,groups,directory,signature,device,steps=STEPS):
    directory.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.98),weight_decay=.01)
    generator=torch.Generator().manual_seed(42)
    last,best=directory/'checkpoint-last.pt',directory/'checkpoint-best.pt'
    step,real,supervised,elapsed,history,bestscore=0,0,0,0.,[],float('inf')
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=True);assert saved['signature']==signature
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        step,real,supervised,elapsed,history,bestscore=(saved[k] for k in ['step','real_tokens','supervised_tokens','wall_seconds','history','best_score'])
        del saved;print('REPRISE_S2S',directory.name,step,flush=True)
    sync();started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    def save(done):
        sync();payload={'signature':signature,'config':asdict(model.cfg),'step':done,'target_steps':steps,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'python_rng':random.getstate(),'real_tokens':real,'supervised_tokens':supervised,'wall_seconds':elapsed+time.perf_counter()-started,'history':history,'best_score':bestscore,'precision':'bfloat16 autocast, FP32 weights/AdamW, no scaler needed','scheduler':'200-step warmup then cosine to 10% LR; phase0 500 denoising steps'}
        atomic_torch(last,payload);atomic_json(directory/'progress.json',{k:payload[k] for k in ['step','target_steps','real_tokens','supervised_tokens','wall_seconds','best_score','history']})
    if step==0:save(0)
    for i in range(step,steps):
        model.train();optimizer.zero_grad(set_to_none=True)
        ratio=(i+1)/WARMUP if i<WARMUP else .1+.9*.5*(1+math.cos(math.pi*(i-WARMUP)/(steps-WARMUP)))
        for paramgroup in optimizer.param_groups:paramgroup['lr']=LR*ratio
        if i<500:
            specs=[('monolingual',16,1.)]
        else:
            # Même séquence de tirages et de masques pour les deux variantes.
            domains=['everyday','software','gospel'];domain=domains[(i-500)%3]
            specs=[(domain,16,.5),('lexical',8,.2),('reply',4,.1),('monolingual',8,.2)]
        objective=0.
        for domain,n,weight in specs:
            examples=choose(groups['train'][domain],n,generator)
            s,t,y,ntokens=batch(examples,device,generator,domain=='monolingual')
            with context():value,tokens=loss(model,s,t,y,.05)
            if not torch.isfinite(value):raise FloatingPointError('Perte non finie.')
            (weight*value/tokens).backward();objective+=weight*float(value.detach())/tokens;real+=ntokens;supervised+=tokens
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (i+1)%100==0 or i==step:print('S2S_STEP',directory.name,i+1,'/',steps,'loss',round(objective,4),flush=True)
        if (i+1)%250==0 and i>=500 or i+1==steps:
            ev=evaluate(model,groups['validation'],device);score=ev['selection_score']
            history.append({'step':i+1,'train_objective':objective,'validation':ev,'gradient_norm':float(norm)})
            print('S2S_VALIDATION',directory.name,i+1,json.dumps(ev),flush=True)
            if score<bestscore:
                bestscore=score;atomic_torch(best,{'signature':signature,'step':i+1,'config':asdict(model.cfg),'model':model.state_dict(),'score':score})
        if (i+1)%100==0 or i+1==steps:save(i+1)
    sync();metrics={'steps':steps,'real_tokens':real,'supervised_tokens':supervised,'wall_seconds_including_drive_io':elapsed+time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'best_validation_score':bestscore,'history':history}
    if step>=steps and (directory/'train-metrics.json').exists():metrics=json.loads((directory/'train-metrics.json').read_text())
    selected=torch.load(best,map_location='cpu',weights_only=True);model.load_state_dict(selected['model']);metrics['best_step']=selected['step']
    atomic_json(directory/'train-metrics.json',metrics);del optimizer,selected
    model.zero_grad(set_to_none=True)
    return metrics

@torch.inference_mode()
def score_generation(model,items,tok,device,max_tokens=96):
    from sacrebleu.metrics import CHRF
    output=[]
    for r in items:
        with context():ids,_=generate(model,r['source'],1,2,max_tokens)
        output.append({'id':r['row']['id'],'prompt_fr':r['row']['prompt'],'reference_bci':r['row']['answer'],'generated_bci':tok.decode(ids),'eos':2 in ids,'domain':r['row']['domain'],'native_validated':False})
    m=CHRF(word_order=2);score=m.corpus_score([r['generated_bci'] for r in output],[[r['reference_bci'] for r in output]])
    return {'rows':len(output),'chrF_pp':score.score,'exact_match':sum(canonical(r['generated_bci'])==canonical(r['reference_bci']) for r in output)/len(output),'no_eos':sum(not r['eos'] for r in output),'metric_signature':str(m.get_signature())},output

@torch.inference_mode()
def benchmark(model,example,device):
    model.eval();source=torch.tensor([example['source']],device=device)
    tokens=64;times=[];prefills=[]
    for repeat in range(6):
        sync();start=time.perf_counter()
        with context():memory,valid=model.encode(source)
        sync();prefill=time.perf_counter()-start;states=None;x=torch.tensor([[1]],device=device)
        sync();start=time.perf_counter()
        with context():
            for offset in range(tokens):
                logits,states=model.decode(x,memory,valid,states,offset,True);x=logits[:,-1].argmax(-1,keepdim=True)
        sync();elapsed=time.perf_counter()-start
        if repeat:times.append(elapsed);prefills.append(prefill)
    times.sort();prefills.sort()
    return {'forced_tokens':tokens,'median_decode_seconds':times[2],'tokens_per_second':tokens/times[2],'median_encoder_seconds':prefills[2],'cache_bytes_after_64':state_bytes(states),'source_tokens':source.shape[1],'warmup':1,'repeats':5,'precision':'bf16','cache_scope':'decoder self+cross K/V, excludes encoder hidden memory'}

def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert (ROOT/'parallel-prepare.jsonl').exists()
    device=torch.device('cuda:0');torch.set_num_threads(4)
    tok,groups=prepare()
    code=pathlib.Path(__file__).parent
    fingerprint={'code':{p.name:sha(p) for p in code.glob('*.py')},'data':{p.name:sha(p) for p in [ROOT/'parallel-prepare.jsonl',ROOT/'replies-prepare.jsonl',ROOT/'tokenizer.json',ROOT/'data/waxal-baoule-text.jsonl']},'seed':42,'steps':STEPS,'selection':'unweighted mean of the four domain validation CE losses; no test-based selection','initialization':'random new S2S weights; v0 checkpoint preserved; incompatible architecture, no weights transferred'}
    atomic_json(ROOT/'EXPERIMENT.json',fingerprint)
    report={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'seed':42,'models':{},'scope':'Research training. Direct/pivot translations plus three trained reply intents, not general dialogue; no native validation.'}
    for variant,ffn in [('hybrid',768),('attention_reference',728)]:
        print('=== S2S',variant,'===',flush=True)
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
        cfg=S2SConfig(vocab_size=tok.get_vocab_size(),variant='hybrid' if variant=='hybrid' else 'attention',ffn_dim=ffn)
        model=MandiaS2S(cfg).to(device);directory=ROOT/variant
        desc=model.description();signature=hashlib.sha256(json.dumps({'experiment':fingerprint,'config':asdict(cfg)},sort_keys=True).encode()).hexdigest()
        trained=train(model,groups,directory,signature,device)
        test=evaluate(model,groups['test'],device)
        generations={};alloutputs=[]
        for domain in ['everyday','software','gospel','lexical']:
            # Sélection fixe des 32 premiers IDs lexicalement triés, identique aux 2 modèles.
            examples=sorted(groups['test'][domain],key=lambda x:x['row']['id'])[:32]
            metric,out=score_generation(model,examples,tok,device);generations[domain]=metric;alloutputs+=out
        demos,demout=score_generation(model,groups['train']['reply'],tok,device,24)
        probes=[]
        for prompt,answer in [('Salut Mandia !','Anouho'),('Bonjour à toi','Anouho'),('Est-ce que tu vas bien ?','mi wun ti kpa'),('Tu te portes bien ?','mi wun ti kpa'),('Je dois partir, au revoir','ye tɛ wo nu'),('Au revoir et à bientôt','ye tɛ wo nu')]:
            probes.append({'source':[4]+tok.encode(prompt).ids+[2],'row':{'id':'probe-'+str(len(probes)),'prompt':prompt,'answer':answer,'domain':'reply_paraphrase_probe'}})
        probemetric,probeout=score_generation(model,probes,tok,device,24)
        perf=benchmark(model,groups['test']['everyday'][0],device)
        # Diagnostic de dépendance au français, même padding après permutation des sources.
        items=groups['test']['everyday'][:32];s,t,y,_=batch(items,device)
        with torch.inference_mode(),context():matched,n=loss(model,s,t,y);wrong,_=loss(model,s.roll(1,0),t,y)
        dependence={'examples':len(items),'matched_loss':float(matched)/n,'permuted_source_loss':float(wrong)/n,'loss_gap':float(wrong-matched)/n,'scope':'Diagnostic only, not native semantic correctness.'}
        metrics={'description':desc,'training':trained,'test_all':test,'generation_test':generations,'training_reply_demos':demos,'reply_paraphrase_probes':probemetric,'performance':perf,'source_dependence':dependence}
        atomic_json(directory/'metrics.json',metrics);atomic_json(directory/'test-generations.json',alloutputs);atomic_json(directory/'reply-demos.json',{'scope':'Seen training prompts, not an independent evaluation','rows':demout,'paraphrase_probes':probeout})
        atomic_torch(directory/'model.pt',{'config':asdict(cfg),'state_dict':model.state_dict(),'tokenizer_sha256':sha(ROOT/'tokenizer.json')})
        report['models'][variant]=metrics;atomic_json(ROOT/'rapport-etape4.json',report)
        print('S2S_RESULTAT',variant,json.dumps({k:v for k,v in metrics.items() if k!='training'}),flush=True)
        print('S2S_REPLIES',variant,json.dumps({'training':demout,'paraphrase_probes':probeout},ensure_ascii=False),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    a,b=report['models'].values();assert a['training']['real_tokens']==b['training']['real_tokens'];assert a['training']['supervised_tokens']==b['training']['supervised_tokens']
    report['same_token_budget_verified']=True
    winner=min(report['models'],key=lambda k:report['models'][k]['training']['best_validation_score'])
    report['candidate_by_validation']=winner
    report['limitations']=['one seed','same tokens, not same GPU time','no native speaker evaluation','GATITOS French uses an English pivot','limited trained reply intents','Bible audio/text versions identified but not full imported; misleading HF bci-bible excluded','published primitives, no novelty demonstrated']
    atomic_json(ROOT/'rapport-etape4.json',report)
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__']) and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
    receipt={'root':str(ROOT),'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],'verified_on_mounted_drive':True,'candidate_by_validation':winner}
    atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',receipt)
    print('MANDIA_S2S_REPORT_BEGIN',json.dumps(report,ensure_ascii=False),'MANDIA_S2S_REPORT_END',flush=True)
    print('MANDIA_S2S_SAVE_BEGIN',json.dumps(receipt),'MANDIA_S2S_SAVE_END',flush=True)
    return report
if __name__=='__main__':main()
