"""Ablation CE vs CE + source contrastive hinge, fixed own 13.78M model.
All checkpoints on mounted Drive. Small written-dialogue pilot, not native validation.
"""
import json,pathlib,hashlib,random,math,time,tempfile,gc,copy
from dataclasses import asdict
import torch
from torch.nn import functional as F
import mandia_stage4 as s4
import mandia_stage5 as s5
from mandia_seq2seq import MandiaS2S,S2SConfig
ROOT=s4.ROOT.parent/'mandia-compact-s2s-etape6'
STEPS,LR=800,3e-5

def encode(rows,tok):
 out={s:{t:[] for t in ['translate','reply']} for s in ['train','validation','test']};excluded=[]
 for r in rows:
  source=[3 if r['task']=='translate' else 4]+tok.encode(r['prompt']).ids+[2]
  target=tok.encode(r['answer']).ids+[2]
  if len(source)>160 or len(target)>192:excluded.append({'id':r['id'],'source':len(source),'target':len(target)});continue
  out[r['split']][r['task']].append({'source':source,'target':target,'row':r})
 assert all(out[s][t] for s in out for t in out[s])
 return out,excluded

def row_nll(logits,labels):
 values=F.cross_entropy(logits.float().transpose(1,2),labels,ignore_index=-100,reduction='none')
 return values.sum(1)/(labels!=-100).sum(1).clamp_min(1)

def paired_loss(model,items,device):
 a,b,y,real=s4.batch(items,device)
 # Both arms execute both forwards and backwards; only the contrastive weight differs.
 logits=model(a,b);ce=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100,label_smoothing=.05,reduction='mean')
 positive=row_nll(logits,y);negative=row_nll(model(a.roll(1,0),b),y)
 previous=[items[-1]]+items[:-1]
 valid=torch.tensor([r['row']['source_document']!=p['row']['source_document'] and s4.canonical(r['row']['answer'])!=s4.canonical(p['row']['answer']) and s4.canonical(r['row']['prompt'])!=s4.canonical(p['row']['prompt']) for r,p in zip(items,previous)],device=device)
 hinge=(F.relu(.2+positive-negative)*valid).sum()/valid.sum().clamp_min(1)
 return ce,hinge,real,int((y!=-100).sum()),int(valid.sum()),int((a!=0).sum()+ (y!=-100).sum())

@torch.inference_mode()
def losses(model,groups,device):
 model.eval();metrics={}
 for task,items in groups.items():
  value=tokens=0
  for start in range(0,len(items),16):
   a,b,y,_=s4.batch(items[start:start+16],device)
   with s4.context():v,n=s4.loss(model,a,b,y)
   value+=float(v);tokens+=n
  metrics[task]={'loss':value/tokens,'tokens':tokens,'rows':len(items)}
 return {'loss':sum(x['loss'] for x in metrics.values())/len(metrics),'tasks':metrics}

@torch.inference_mode()
def source_diagnostic(model,items,device):
 model.eval();a,b,y,_=s4.batch(items,device)
 with s4.context():p=row_nll(model(a,b),y);n=row_nll(model(a.roll(1,0),b),y)
 return {'rows':len(items),'matched_nll':float(p.mean()),'permuted_nll':float(n.mean()),'gap':float((n-p).mean()),'scope':'source dependence diagnostic, not semantic or native correctness'}

def train(model,specs,evaluate,directory,signature,device,contrastive_weight,steps=STEPS,every=100,interrupt_at=None):
 directory.mkdir(parents=True,exist_ok=True);last=directory/'checkpoint-last.pt';best=directory/'checkpoint-best.pt'
 optimizer=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.98),weight_decay=.01)
 sampler=torch.Generator().manual_seed(42)
 step=real=supervised=negative_tokens=valid_pairs=0;elapsed=0.;history=[];bestscore=float('inf');baseline=None
 if last.exists():
  c=torch.load(last,map_location='cpu',weights_only=True);assert c['signature']==signature
  model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);sampler.set_state(c['sampler_rng']);torch.set_rng_state(c['torch_rng']);random.setstate(c['python_rng'])
  if device.type=='cuda':torch.cuda.set_rng_state_all(c['cuda_rng'])
  step,real,supervised,negative_tokens,valid_pairs,elapsed,history,bestscore,baseline=(c[k] for k in ['step','real_tokens','supervised_tokens','negative_forward_tokens','valid_negative_pairs','wall_seconds','history','best_score','baseline_validation'])
  del c;print('ETAPE6_REPRISE',directory.name,step,flush=True)
 s5.sync(device);start=time.perf_counter()
 def save(done):
  s5.sync(device)
  c={'signature':signature,'config':asdict(model.cfg),'step':done,'target_steps':steps,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler_rng':sampler.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'python_rng':random.getstate(),'real_tokens':real,'supervised_tokens':supervised,'negative_forward_tokens':negative_tokens,'valid_negative_pairs':valid_pairs,'wall_seconds':elapsed+time.perf_counter()-start,'history':history,'best_score':bestscore,'baseline_validation':baseline}
  s4.atomic_torch(last,c);s4.atomic_json(directory/'progress.json',{k:c[k] for k in ['step','target_steps','real_tokens','supervised_tokens','negative_forward_tokens','valid_negative_pairs','wall_seconds','best_score']})
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
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  if (i+1)%50==0:print('ETAPE6_STEP',directory.name,i+1,'/',steps,round(objective,4),'hinge',round(hinge_value,4),flush=True)
  if (i+1)%every==0 or i+1==steps:
   ev=evaluate(model,baseline);history.append({'step':i+1,'objective':objective,'hinge':hinge_value,'validation':ev});print('ETAPE6_VALIDATION',directory.name,i+1,json.dumps(ev),flush=True)
   if ev['eligible'] and ev['selection_score']<bestscore:
    bestscore=ev['selection_score'];s4.atomic_torch(best,{'model':model.state_dict(),'step':i+1,'score':bestscore})
  if (i+1)%100==0 or i+1==steps or i+1==interrupt_at:save(i+1)
  if i+1==interrupt_at:return None
 chosen=torch.load(best,map_location='cpu',weights_only=True);model.load_state_dict(chosen['model']);model.zero_grad(set_to_none=True)
 return {'best_step':chosen['step'],'best_score':bestscore,'steps':steps,'real_tokens':real,'supervised_tokens':supervised,'negative_forward_tokens':negative_tokens,'valid_negative_pairs':valid_pairs,'wall_seconds':elapsed+time.perf_counter()-start,'history':history}

def verify():
 previous=s4.DEVICE;s4.DEVICE=torch.device('cpu')
 try:
  cfg=S2SConfig(vocab_size=32,dim=48,heads=6,kv_heads=2,encoder_layers=1,decoder_layers=2,ffn_dim=64,dropout=.1)
  items=[{'source':[4,8+i,2],'target':[12+i,2],'row':{'source_document':str(i),'prompt':str(i),'answer':str(i)}} for i in range(4)]
  specs=lambda i:[(items,4,1.,True)]
  ev=lambda m,b:{'selection_score':2.,'eligible':True}
  with tempfile.TemporaryDirectory() as td:
   td=pathlib.Path(td)
   for name in ['full','resume']:
    torch.manual_seed(19);random.seed(19);model=MandiaS2S(cfg)
    train(model,specs,ev,td/name,'resume-fixture',s4.DEVICE,.1,steps=8,every=4,interrupt_at=4 if name=='resume' else None)
    if name=='resume':train(MandiaS2S(cfg),specs,ev,td/name,'resume-fixture',s4.DEVICE,.1,steps=8,every=4)
   a=torch.load(td/'full/checkpoint-last.pt',weights_only=True);b=torch.load(td/'resume/checkpoint-last.pt',weights_only=True)
   assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
   for k,values in a['optimizer']['state'].items():
    for field,tensor in values.items():assert torch.equal(tensor,b['optimizer']['state'][k][field])
   for field in ['torch_rng','sampler_rng']:assert torch.equal(a[field],b[field])
   for field in ['python_rng','real_tokens','supervised_tokens','negative_forward_tokens','valid_negative_pairs']:assert a[field]==b[field]
   x=torch.tensor([[[3.,0.,-2.],[0.,3.,-2.]]]);labels=torch.tensor([[0,-100]])
   assert torch.allclose(row_nll(x,labels),F.cross_entropy(x[:,:1].reshape(-1,3),torch.tensor([0])).reshape(1))
  return {'actual_trainer_cpu_resume_bitwise_equal':True,'weights_optimizer_rng_counters':True,'padding_ignored_per_row_nll':True,'cuda_bitwise_not_claimed':True}
 finally:s4.DEVICE=previous

def main():
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'GPU bf16 requis'
 device=torch.device('cuda');s4.DEVICE=device;torch.set_num_threads(4);ROOT.mkdir(exist_ok=True)
 s4.atomic_json(ROOT/'VERIFICATION-REPRISE.json',verify())
 tok,old=s4.prepare()
 earlier,_=s5.encode([json.loads(l) for l in (s5.ROOT/'quotidien-prepare.jsonl').read_text().splitlines()],tok)
 rows=[json.loads(l) for l in (ROOT/'dialogues-prepare.jsonl').read_text().splitlines()];new,excluded=encode(rows,tok)
 parent=s5.ROOT/'model.pt';assert s4.sha(parent)=='b19182b18a271dc7a6b36abc60ba07dfbb5848e3f020f7c61e2bf3eedf46042e'
 payload=torch.load(parent,map_location='cpu',weights_only=True);assert payload['tokenizer_sha256']==s4.sha(s4.ROOT/'tokenizer.json')
 experiment={'date':'2026-10-14','parent_model_sha256':s4.sha(parent),'tokenizer_sha256':payload['tokenizer_sha256'],'data_sha256':s4.sha(ROOT/'dialogues-prepare.jsonl'),'code_sha256':s4.sha(__file__),'stage4_code_sha256':s4.sha(s4.__file__),'stage5_code_sha256':s4.sha(s5.__file__),'steps_per_arm':STEPS,'lr':LR,'seed':42,'initialization':'own selected stage5 weights, fresh AdamW per arm','counts':{s:{t:len(v) for t,v in g.items()} for s,g in new.items()},'excluded_length':excluded,'mix':'.4 new actual next-turn reply; .2 new translation; .15 stage5 train; .2 stage4 domain replay; .05 trained 14 replies','contrastive':'lambda .1 vs 0; max(0,.2 + mean matched NLL - mean permuted-source NLL); different document/prompt/answer mask; same extra negative forward+backward in both arms for controlled ablation','validation_selection':'.35 new translate CE + .35 new reply CE + .15 stage5 CE + .15 old balanced CE; old CE <= initial+.10, stage5 CE <= initial+.15, all 14 old replies retained; arm chosen before test','native_validation':False,'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}
 s4.atomic_json(ROOT/'EXPERIMENT.json',experiment)
 def validate(m,base):
  n=losses(m,new['validation'],device);prior=s5.new_loss(m,earlier['validation'],device);o=s4.evaluate(m,old['validation'],device);d,_=s5.demo(m,old,tok,device)
  return {'new':n,'prior_validation':prior['loss'],'old_validation':o['selection_score'],'old_reply_exact':d['exact_match'],'eligible':d['exact_match']==1 and (base is None or o['selection_score']<=base['old_validation']+.10 and prior['loss']<=base['prior_validation']+.15),'selection_score':.7*n['loss']+.15*prior['loss']+.15*o['selection_score']}
 def specs(i):
  domain=['everyday','software','gospel','lexical'][i%4];task='translate' if i%2==0 else 'guided'
  return [(new['train']['reply'],8,.4,True),(new['train']['translate'],8,.2,False),(earlier['train'][task],8,.15,False),(old['train'][domain],8,.2,False),(old['train']['reply'],4,.05,False)]
 def load_parent():
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
  m=MandiaS2S(S2SConfig(**payload['config'])).to(device);m.load_state_dict(payload['state_dict']);return m
 # Arms and winner fixed using validation only; no test results are computed until selection is frozen.
 arms={}
 for name,weight in [('ce_control',0.),('source_contrastive',.1)]:
  m=load_parent();directory=ROOT/name
  signature=hashlib.sha256(json.dumps({'experiment':experiment,'arm':name,'weight':weight},sort_keys=True).encode()).hexdigest()
  trained=train(m,specs,validate,directory,signature,device,weight)
  s4.atomic_torch(directory/'model.pt',{'config':payload['config'],'state_dict':m.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256']})
  arms[name]={'training':trained,'contrastive_weight':weight};s4.atomic_json(directory/'train-report.json',arms[name])
  del m;gc.collect();torch.cuda.empty_cache()
 assert len({x['training']['real_tokens'] for x in arms.values()})==1
 assert len({x['training']['supervised_tokens'] for x in arms.values()})==1
 assert len({x['training']['negative_forward_tokens'] for x in arms.values()})==1
 chosen=min(arms,key=lambda name:arms[name]['training']['best_score'])
 s4.atomic_json(ROOT/'SELECTION-AVANT-TEST.json',{'winner_by_validation':chosen,'scores':{k:v['training']['best_score'] for k,v in arms.items()},'test_not_used':True})
 def test(m):
  result={'new_test':losses(m,new['test'],device),'old_test':s4.evaluate(m,old['test'],device),'prior_test':s5.new_loss(m,earlier['test'],device),'old_reply':s5.demo(m,old,tok,device)[0],'generation':{},'source_dependence':source_diagnostic(m,new['test']['reply'],device)};outputs=[]
  for task,items in new['test'].items():
   result['generation'][task],out=s4.score_generation(m,sorted(items,key=lambda x:x['row']['id']),tok,device,96);outputs.extend(dict(x,task=task) for x in out)
  return result,outputs
 m=load_parent();before,outputs=test(m);s4.atomic_json(ROOT/'generations-avant.json',outputs);del m;gc.collect();torch.cuda.empty_cache()
 for name in arms:
  m=load_parent();c=torch.load(ROOT/name/'model.pt',map_location='cpu',weights_only=True);m.load_state_dict(c['state_dict']);del c
  arms[name]['test'],outputs=test(m);s4.atomic_json(ROOT/name/'generations-test.json',outputs)
  del m;gc.collect();torch.cuda.empty_cache()
 report={'experiment':experiment,'before':before,'arms':arms,'selected_by_validation':chosen,'same_sampling_token_budget_verified':True,'parameters':sum(v.numel() for v in payload['state_dict'].values()),'limits':['8 written/synthetic source documents, only 1 validation and 2 test documents','French translated by assistant from English; not human reviewed','Published Baoule targets show spelling variation; no native validation','Contrastive negative pairs can still be semantically compatible','This tests training objective at fixed architecture, not architectural novelty','New test was opened only after validation-based selection; prior stage5 test already observed previously','Additional negative passes deliberately included in both arms for compute control; not a deployment timing benchmark']}
 s4.atomic_json(ROOT/'rapport-etape6.json',report)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
 s4.atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',{'verified_on_mounted_drive':True,'selected_by_validation':chosen,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':s4.sha(p)} for p in sorted(files)]})
 print('MANDIA_ETAPE6_TERMINE',json.dumps({'winner':chosen,'before':before,'after':{k:v['test'] for k,v in arms.items()}},ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':main()
