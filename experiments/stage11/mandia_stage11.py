"""Stage11: controlled CE / source contrast / source contrast + offline sequence UL.
All arms same data/sampler/auxiliary forwards, own 20.86M parent, Drive checkpoints.
UL negatives are generated from TRAIN sources only; no invented positive Baoule.
"""
import pathlib,json,hashlib,random,math,time,tempfile,gc,re,unicodedata
from dataclasses import asdict
import torch
from torch.nn import functional as F
import mandia_stage4 as s4
import mandia_stage5 as s5
import mandia_stage6 as s6
import mandia_stage7 as s7
import mandia_stage10 as s10
from mandia_seq2seq import MandiaS2S,S2SConfig,generate
ROOT=s4.ROOT.parent/'mandia-compact-s2s-etape11'
STEPS,LR=300,1e-5
paired_loss=s6.paired_loss

def completion_mask(generated,reference,n=8):
 gold={tuple(reference[i:i+n]) for i in range(len(reference)-n+1)};seen=set();mask=[]
 for i,t in enumerate(generated):
  g=tuple(generated[i-n+1:i+1]) if i>=n-1 else ()
  mask.append(bool(g and g in seen and g not in gold and t>=5))
  if g:seen.add(g)
 return mask

def unlikelihood_loss(model,items,device):
 a,b,y,real=s4.batch(items,device);logits=model(a,b).float()
 mask=torch.zeros_like(y,dtype=torch.bool)
 for k,x in enumerate(items):mask[k,:len(x['ul_mask'])]=torch.tensor(x['ul_mask'],device=device)
 mask &= y.ge(5)
 probs=logits.softmax(-1).gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
 values=-torch.log1p(-probs.clamp(max=1-1e-6))
 return (values*mask).sum()/mask.sum().clamp_min(1),int(mask.sum()),real

def repeated4(text):
 words=re.findall(r"[^\W_]+(?:['’][^\W_]+)*",unicodedata.normalize('NFC',text).casefold())
 grams=[tuple(words[i:i+4]) for i in range(len(words)-3)]
 return (len(grams)-len(set(grams)))/max(1,len(grams))

def encode(rows,tok):
 result=[]
 for r in rows:
  x={'source':[4 if r['task']=='reply' else 3]+tok.encode(r['prompt']).ids+[2], 'target':tok.encode(r['answer']).ids+[2],'row':r}
  assert max(len(x['source']),len(x['target']))<=512
  result.append(x)
 return result

@torch.inference_mode()
def diagnostic(m,items,device):
 m.eval();a,b,y,_=s4.batch(items,device)
 with s4.context():positive=s6.row_nll(m(a,b),y);negative=s6.row_nll(m(a.roll(1,0),b),y)
 previous=[items[-1]]+items[:-1]
 mask=torch.tensor([x['row']['source_document']!=z['row']['source_document'] and s4.canonical(x['row']['answer'])!=s4.canonical(z['row']['answer']) for x,z in zip(items,previous)],device=device)
 assert mask.any()
 return {'rows':len(items),'valid_pairs':int(mask.sum()),'matched_nll':float(positive[mask].mean()),'permuted_nll':float(negative[mask].mean()),'gap':float((negative-positive)[mask].mean()),'hinge':float(F.relu(.2+positive[mask]-negative[mask]).mean()),'scope':'different document source permutation; diagnostic only, not proof of comprehension'}

@torch.inference_mode()
def generation(m,items,tok,device,cap=192):
 metrics,out=s4.score_generation(m,items,tok,device,cap)
 rep=[repeated4(r['generated_bci']) for r in out];gold=[repeated4(r['reference_bci']) for r in out]
 metrics.update(mean_repeated_word4=sum(rep)/len(rep),mean_reference_repeated_word4=sum(gold)/len(gold),mean_excess_repeated_word4=sum(max(0,a-b) for a,b in zip(rep,gold))/len(rep),repetitive_outputs=sum(v>.1 for v in rep),empty_outputs=sum(not r['generated_bci'].strip() for r in out),mean_generated_tokens=sum(len(tok.encode(r['generated_bci']).ids) for r in out)/len(out),mean_reference_tokens=sum(len(x['target'])-1 for x in items)/len(items))
 for r,a,b in zip(out,rep,gold):r.update(repeated_word4=a,reference_repeated_word4=b)
 return metrics,out

def train(model,specs,evaluate,directory,signature,device,contrastive_weight,steps=STEPS,every=50,interrupt_at=None,negative_pool=None,ul_weight=0.):
 directory.mkdir(parents=True,exist_ok=True);last=directory/'checkpoint-last.pt';best=directory/'checkpoint-best.pt'
 optimizer=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.98),weight_decay=.01)
 sampler=torch.Generator().manual_seed(42)
 step=real=supervised=negative_tokens=valid_pairs=ul_tokens=ul_forward_tokens=0;elapsed=0.;history=[];bestscore=float('inf');baseline=None
 if last.exists():
  c=torch.load(last,map_location='cpu',weights_only=True);assert c['signature']==signature
  model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);sampler.set_state(c['sampler_rng']);torch.set_rng_state(c['torch_rng']);random.setstate(c['python_rng'])
  if device.type=='cuda':torch.cuda.set_rng_state_all(c['cuda_rng'])
  step,real,supervised,negative_tokens,valid_pairs,elapsed,history,bestscore,baseline=(c[k] for k in ['step','real_tokens','supervised_tokens','negative_forward_tokens','valid_negative_pairs','wall_seconds','history','best_score','baseline_validation'])
  ul_tokens=c['ul_penalized_tokens'];ul_forward_tokens=c['ul_forward_tokens']
  del c;print('ETAPE11_REPRISE',directory.name,step,flush=True)
 s5.sync(device);start=time.perf_counter()
 def save(done):
  s5.sync(device)
  c={'signature':signature,'config':asdict(model.cfg),'step':done,'target_steps':steps,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler_rng':sampler.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'python_rng':random.getstate(),'real_tokens':real,'supervised_tokens':supervised,'negative_forward_tokens':negative_tokens,'valid_negative_pairs':valid_pairs,'wall_seconds':elapsed+time.perf_counter()-start,'history':history,'best_score':bestscore,'baseline_validation':baseline,'ul_penalized_tokens':ul_tokens,'ul_forward_tokens':ul_forward_tokens}
  s4.atomic_torch(last,c);s4.atomic_json(directory/'progress.json',{k:c[k] for k in ['step','target_steps','real_tokens','supervised_tokens','negative_forward_tokens','valid_negative_pairs','wall_seconds','best_score','ul_penalized_tokens','ul_forward_tokens']})
 if step==0:
  baseline=evaluate(model,None);bestscore=baseline['selection_score'];history.append({'step':0,'validation':baseline})
  s4.atomic_torch(best,{'model':model.state_dict(),'step':0,'score':bestscore});save(0)
 for i in range(step,steps):
  model.train();optimizer.zero_grad(set_to_none=True);objective=0.;hinge_value=0.
  ratio=(i+1)/40 if i<40 else .2+.8*.5*(1+math.cos(math.pi*(i-40)/max(1,steps-40)))
  for pg in optimizer.param_groups:pg['lr']=LR*ratio
  for items,n,weight,is_dialogue in specs(i):
   selected=s4.choose(items,n,sampler)
   if is_dialogue:
    with s4.context():ce,hinge,nreal,ntarget,nvalid,nnegative=paired_loss(model,selected,device);v=weight*ce+contrastive_weight*hinge
    negative_tokens+=nnegative;valid_pairs+=nvalid;hinge_value=float(hinge.detach())
   else:
    a,b,y,nreal=s4.batch(selected,device)
    with s4.context():summed,ntarget=s4.loss(model,a,b,y,.05);v=weight*summed/ntarget
   if not torch.isfinite(v):raise FloatingPointError('Non-finite objective')
   v.backward();objective+=float(v.detach());real+=nreal;supervised+=ntarget
  if negative_pool:
   selected=s4.choose(negative_pool,4,sampler)
   with s4.context():u,marked,work=unlikelihood_loss(model,selected,device)
   (ul_weight*u).backward();ul_tokens+=marked;ul_forward_tokens+=work
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  if (i+1)%50==0:print('ETAPE11_STEP',directory.name,i+1,'/',steps,round(objective,4),'hinge',round(hinge_value,4),flush=True)
  if (i+1)%every==0 or i+1==steps:
   ev=evaluate(model,baseline);history.append({'step':i+1,'objective':objective,'hinge':hinge_value,'validation':ev});print('ETAPE11_VALIDATION',directory.name,i+1,json.dumps(ev),flush=True)
   if ev['eligible'] and ev['selection_score']<bestscore:
    bestscore=ev['selection_score'];s4.atomic_torch(best,{'model':model.state_dict(),'step':i+1,'score':bestscore})
  if (i+1)%100==0 or i+1==steps or i+1==interrupt_at:save(i+1)
  if i+1==interrupt_at:return None
 chosen=torch.load(best,map_location='cpu',weights_only=True);model.load_state_dict(chosen['model']);model.zero_grad(set_to_none=True)
 return {'best_step':chosen['step'],'best_score':bestscore,'steps':steps,'real_tokens':real,'supervised_tokens':supervised,'negative_forward_tokens':negative_tokens,'valid_negative_pairs':valid_pairs,'wall_seconds':elapsed+time.perf_counter()-start,'history':history,'ul_penalized_tokens':ul_tokens,'ul_forward_tokens':ul_forward_tokens}

def verify():
 previous=s4.DEVICE;s4.DEVICE=torch.device('cpu')
 try:
  assert not any(completion_mask([5,6,7,8,9,10,11,12]*3,[5,6,7,8,9,10,11,12]*3))
  assert sum(completion_mask([5,6,7,8,9,10,11,12]*3,[20,21]))==9
  logits=torch.tensor([[[0.,1.,2.]]],requires_grad=True);prob=logits.softmax(-1)[0,0,2];(-torch.log1p(-prob)).backward();assert logits.grad[0,0,2]>0
  cfg=S2SConfig(vocab_size=32,dim=48,heads=6,kv_heads=2,encoder_layers=1,decoder_layers=2,ffn_dim=64,dropout=.1)
  items=[{'source':[3,8+i,2],'target':[12+i,2],'row':{'source_document':str(i),'prompt':str(i),'answer':str(i)}} for i in range(4)]
  neg=[dict(x,target=[12,13,12,13,2],ul_mask=[False,False,True,True,False]) for x in items]
  fixture=MandiaS2S(cfg)
  _,marks,_=unlikelihood_loss(fixture,[dict(items[0],target=[12,2],ul_mask=[True,True]),dict(items[1],target=[13,14,2],ul_mask=[True,False,True])],s4.DEVICE);assert marks==2
  specs=lambda i:[(items,4,1.,True)];ev=lambda m,b:{'selection_score':2.,'eligible':True}
  with tempfile.TemporaryDirectory() as td:
   td=pathlib.Path(td)
   for name in ['full','resume']:
    torch.manual_seed(19);random.seed(19);m=MandiaS2S(cfg)
    train(m,specs,ev,td/name,'s11-resume-fixture',s4.DEVICE,.3,steps=8,every=4,interrupt_at=4 if name=='resume' else None,negative_pool=neg,ul_weight=.1)
    if name=='resume':train(MandiaS2S(cfg),specs,ev,td/name,'s11-resume-fixture',s4.DEVICE,.3,steps=8,every=4,negative_pool=neg,ul_weight=.1)
   a=torch.load(td/'full/checkpoint-last.pt',weights_only=True);b=torch.load(td/'resume/checkpoint-last.pt',weights_only=True)
   assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
   for k,vs in a['optimizer']['state'].items():
    for f,t in vs.items():assert torch.equal(t,b['optimizer']['state'][k][f])
   for f in ['torch_rng','sampler_rng']:assert torch.equal(a[f],b[f])
   for f in ['python_rng','real_tokens','supervised_tokens','negative_forward_tokens','valid_negative_pairs','ul_penalized_tokens','ul_forward_tokens']:assert a[f]==b[f]
  return {'actual_stage11_trainer_cpu_resume_bitwise_equal':True,'weights_optimizer_rng_and_counters':True,'gold_8grams_protected':True,'repeated_8gram_completions_masked':True,'ul_gradient_lowers_negative_probability':True,'padding_eos_special_tokens_excluded':True,'cuda_bitwise_not_claimed':True}
 finally:s4.DEVICE=previous

def main():
 assert pathlib.Path('/content/drive/MyDrive').exists() and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 device=torch.device('cuda');s4.DEVICE=device;torch.set_num_threads(4)
 ROOT.mkdir(exist_ok=True);s4.atomic_json(ROOT/'VERIFICATION-REPRISE-ET-PERTES.json',verify());torch.set_num_threads(4)
 tok,old=s4.prepare();new=s10.encode(s10.read(s10.ROOT/'nouveaux-paralleles.jsonl'),tok)
 p6,_=s6.encode(s10.read(s6.ROOT/'dialogues-prepare.jsonl'),tok)
 r9=ROOT.parent/'mandia-compact-s2s-etape9';written=s7.encode(s10.read(r9/'dialogues-ecrits-ajout.jsonl'),tok)
 daily=s7.encode(s10.read(s7.ROOT/'teacher-grounded-qa.jsonl'),tok)
 parallel=new['train']['sentences']+p6['train']['translate'];replies=new['train']['reply']+p6['train']['reply']+written['train']
 validation_source=[x for x in new['validation']['sentences'] if x['row']['domain']=='new_sentence_translation']+p6['validation']['reply']
 probes=new['validation']['long'][:2]+p6['validation']['reply'][:2]
 parent=s10.ROOT/'translation_dialogue/model.pt';assert s4.sha(parent)=='5d3004982bdc53ea361d7bb23f99ef8800fa1f1163ee01a0388abffa5b357db0'
 payload=torch.load(parent,map_location='cpu',weights_only=True);assert payload['tokenizer_sha256']==s4.sha(s4.ROOT/'tokenizer.json')
 def load_parent():
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
  m=MandiaS2S(S2SConfig(**payload['config'])).to(device);m.load_state_dict(payload['state_dict']);return m
 # Fixed offline negative bank, generated solely from existing TRAIN sources by the frozen parent.
 negative_file=ROOT/'negatifs-entrainement-parent.json'
 if not negative_file.exists():
  m=load_parent();bank=[]
  for x in new['train']['long']+parallel[:24]+replies[:16]:
   with torch.inference_mode(),s4.context():ids,_=generate(m,x['source'],1,2,192)
   mask=completion_mask(ids,x['target'])
   if any(mask):bank.append({'source':x['source'],'target':ids,'ul_mask':mask,'row':x['row'],'parent_generated':tok.decode(ids),'negative_only':True,'positive_reference':x['row']['answer']})
  assert bank,'No repeated train-only negatives: do not manufacture Baoule targets'
  s4.atomic_json(negative_file,bank);del m;gc.collect();torch.cuda.empty_cache()
 bank=json.loads(negative_file.read_text());assert all(x['row']['split']=='train' for x in bank)
 paths=[s10.ROOT/'nouveaux-paralleles.jsonl',s4.ROOT/'parallel-prepare.jsonl',s4.ROOT/'replies-prepare.jsonl',s6.ROOT/'dialogues-prepare.jsonl',r9/'dialogues-ecrits-ajout.jsonl',s7.ROOT/'teacher-grounded-qa.jsonl',ROOT/'nouveaux-tests.jsonl',negative_file]
 experiment={'stage':11,'date':'2026-10-14','parent_sha256':s4.sha(parent),'parameters':20860416,'architecture_changed':False,'seed':42,'steps_per_arm':STEPS,'lr':LR,'optimizer':'fresh AdamW(.9,.98),wd.01,clip1,smoothing.05,warm40cosine20%',
  'arms':{'ce_control':[0.,0.],'source_contrastive':[.3,0.],'source_contrastive_ul':[.3,.1]},
  'negative_bank_rows':len(bank),'masked_negative_tokens':sum(sum(x['ul_mask']) for x in bank),
  'source_objective':'per-group .3 * max(0,.2+matched NLL-permuted-source NLL), valid different documents/prompts/answers; two paired groups',
  'ul':'offline fixed parent TRAIN-only greedy rollouts cap192; penalize completions of repeated8-token grams absent from that gold reference; no special/EOS/pad penalties; .1 mean -log(1-p) on masked positions; no decoding repetition block',
  'mix':'.40 parallel12+.25 replies8+.10 long2+.20 old domain8+.05 trained replies4; both paired groups use source negatives; all arms include auxiliary forwards/backwards, weight0 in controls',
  'selection':'.40 new sentence CE+.25 old SMOL reply CE+.10 new longCE+.15 old balancedCE+.10 authored birthdayQA CE +.5 permutation hinge +.5 excess repeated-word4 on4 fixed validation probes; preserve14, oldCE<=initial+.15, new sentenceCE<=initial+.15, SMOLreplyCE<=initial+.15; step0 allowed; select before any test',
  'same_auxiliary_computation':True,'data_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in paths},'code_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in [pathlib.Path(__file__),pathlib.Path(s7.__file__),pathlib.Path(s10.__file__)]+list(pathlib.Path(s6.__file__).parent.glob('*.py'))},'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
  'limits':['one seed and small continuation, not proof of optimal architecture or efficiency','same positive data and sampled token budgets; dropout and static negatives held fixed','offline parent negatives do not cover new candidate loops','cross-document negatives can be semantically compatible','token repetition heuristic may penalize legitimate phrases despite gold protection','new source documents held out, French assistant-authored; published Baoule without native validation','six written next-turn tests, no newly recorded authentic conversations','previous tests now regression only','automatic source gap/chrF are not human semantic correctness']}
 s4.atomic_json(ROOT/'EXPERIMENT.json',experiment)
 def validate(m,base):
  n=s10.ce(m,new['validation']['sentences'],device);p=s10.ce(m,p6['validation']['reply'],device);l=s10.ce(m,new['validation']['long'],device);o=s4.evaluate(m,old['validation'],device)['selection_score'];q=s10.ce(m,new['validation']['reply'],device)
  d=diagnostic(m,validation_source,device);g,_=generation(m,probes,tok,device,256);r=s5.demo(m,old,tok,device)[0]['exact_match']
  objective=.40*n+.25*p+.10*l+.15*o+.10*q+.5*d['hinge']+.5*g['mean_excess_repeated_word4']
  eligible=r==1 and (base is None or (o<=base['old_ce']+.15 and n<=base['new_sentence_ce']+.15 and p<=base['old_reply_ce']+.15))
  return {'selection_score':objective,'eligible':eligible,'new_sentence_ce':n,'old_reply_ce':p,'long_ce':l,'old_ce':o,'authored_qa_ce':q,'source':d,'probe_generation':g,'trained14':r}
 def specs(i):
  domain=['everyday','software','gospel','lexical'][i%4]
  return [(parallel,12,.40,True),(replies,8,.25,True),(new['train']['long'],2,.10,False),(old['train'][domain],8,.20,False),(old['train']['reply'],4,.05,False)]
 arms={}
 for name,(cw,uw) in experiment['arms'].items():
  m=load_parent();directory=ROOT/name;signature=hashlib.sha256(json.dumps({'experiment':experiment,'arm':name},sort_keys=True).encode()).hexdigest()
  trained=train(m,specs,validate,directory,signature,device,cw,steps=STEPS,every=50,negative_pool=bank,ul_weight=uw)
  s4.atomic_torch(directory/'model.pt',{'config':payload['config'],'state_dict':m.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256']})
  arms[name]={'training':trained,'contrastive_weight':cw,'ul_weight':uw};s4.atomic_json(directory/'train-report.json',arms[name]);del m;gc.collect();torch.cuda.empty_cache()
 for field in ['real_tokens','supervised_tokens','negative_forward_tokens','valid_negative_pairs','ul_penalized_tokens','ul_forward_tokens']:assert len({a['training'][field] for a in arms.values()})==1,field
 chosen=min(arms,key=lambda k:arms[k]['training']['best_score'])
 s4.atomic_json(ROOT/'SELECTION-AVANT-TEST.json',{'selected_by_validation':chosen,'steps':{k:v['training']['best_step'] for k,v in arms.items()},'scores':{k:v['training']['best_score'] for k,v in arms.items()},'test_not_used':True})
 fresh=encode(s10.read(ROOT/'nouveaux-tests.jsonl'),tok)
 groups={k:[x for x in fresh if x['row']['domain']==k] for k in ['fresh_translation','fresh_written_reply','fresh_authored_advice']}
 groups.update(regression_s10_replies=new['test']['reply'],regression_s10_long=new['test']['long'],regression_known40=daily['test'])
 def test(m,label):
  metrics={};outputs={}
  for k,items in groups.items():
   cap=512 if k=='regression_s10_long' else 192
   gen,out=generation(m,items,tok,device,cap);metrics[k]={'ce':s10.ce(m,items,device),'generation':gen};outputs[k]=out
  metrics['source_fresh_translation']=diagnostic(m,groups['fresh_translation'],device)
  metrics['source_fresh_written_reply']=diagnostic(m,groups['fresh_written_reply'],device)
  metrics['old_test_ce']=s4.evaluate(m,old['test'],device)['selection_score'];metrics['trained14']=s5.demo(m,old,tok,device)[0]
  s4.atomic_json(ROOT/f'generations-{label}.json',outputs);return metrics
 m=load_parent();before=test(m,'parent');del m;gc.collect();torch.cuda.empty_cache()
 for name in arms:
  m=load_parent();p=torch.load(ROOT/name/'model.pt',map_location='cpu',weights_only=True);m.load_state_dict(p['state_dict']);del p
  arms[name]['test']=test(m,name);del m;gc.collect();torch.cuda.empty_cache()
 report={'experiment':experiment,'before':before,'arms':arms,'selected_by_validation':chosen,'matched_sampled_token_budgets_verified':True,'limits':experiment['limits']};s4.atomic_json(ROOT/'rapport-etape11.json',report)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
 s4.atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',{'verified_on_mounted_drive':True,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':s4.sha(p)} for p in sorted(files)]})
 print('MANDIA_ETAPE11_TERMINE',json.dumps({'selected':chosen,'steps':{k:v['training']['best_step'] for k,v in arms.items()},'before':before,'after':{k:v['test'] for k,v in arms.items()}},ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':main()
