"""Continuation ciblée du meilleur modèle S2S, corpus nouveau groupé.
600 pas, tokenizer/architecture inchangés ; checkpoints complets dans Drive.
Les réponses guidées demandent une traduction, elles ne sont pas des dialogues libres.
"""
import pathlib,json,hashlib,random,time,math,gc,tempfile,copy
from dataclasses import asdict
import torch
from tokenizers import Tokenizer
import mandia_stage4 as s4
from mandia_seq2seq import MandiaS2S,S2SConfig
ROOT=s4.ROOT.parent/'mandia-compact-s2s-etape5'
STEPS,LR=600,2e-5
def sync(device):
 if device.type=='cuda':torch.cuda.synchronize()
def encode(rows,tok):
 groups={k:{'translate':[],'guided':[]} for k in ['train','validation','test']};excluded=[]
 for r in rows:
  target=tok.encode(r['answer']).ids+[2]
  for task,mode,prompt in [('translate',3,r['prompt']),('guided',4,'Dis en baoulé : '+r['prompt'])]:
   source=[mode]+tok.encode(prompt).ids+[2]
   if len(source)>160 or len(target)>192:excluded.append({'id':r['id'],'task':task});continue
   row=dict(r,prompt=prompt,domain='everyday',task=task)
   groups[r['split']][task].append({'source':source,'target':target,'row':row})
 for split in groups:
  for task in groups[split]:assert groups[split][task],(split,task)
 return groups,excluded
@torch.inference_mode()
def new_loss(model,groups,device):
 model.eval();out={}
 for task,items in groups.items():
  total,tokens=0.,0
  for i in range(0,len(items),16):
   a,b,y,_=s4.batch(items[i:i+16],device)
   with s4.context():v,n=s4.loss(model,a,b,y)
   total+=float(v);tokens+=n
  out[task]={'loss':total/tokens,'rows':len(items),'tokens':tokens}
 return {'loss':sum(x['loss'] for x in out.values())/len(out),'tasks':out}
@torch.inference_mode()
def demo(model,old,tok,device):
 return s4.score_generation(model,old['train']['reply'],tok,device,24)

def train(model,specifications,evaluate,directory,signature,device,steps=STEPS,lr=LR,every=50,save_every=100,interrupt_at=None):
 directory.mkdir(parents=True,exist_ok=True)
 last=directory/'checkpoint-last.pt';best=directory/'checkpoint-best.pt'
 opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=.01)
 g=torch.Generator().manual_seed(42);step=real=supervised=0;elapsed=0.;history=[];bestscore=float('inf');baseline=None
 if last.exists():
  c=torch.load(last,map_location='cpu',weights_only=True);assert c['signature']==signature
  model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);g.set_state(c['sampler_rng']);torch.set_rng_state(c['torch_rng']);random.setstate(c['python_rng'])
  if device.type=='cuda':torch.cuda.set_rng_state_all(c['cuda_rng'])
  step,real,supervised,elapsed,history,bestscore,baseline=(c[k] for k in ['step','real_tokens','supervised_tokens','wall_seconds','history','best_score','baseline_validation']);del c
  print('REPRISE_ETAPE5',step,flush=True)
 sync(device);start=time.perf_counter()
 def save(done):
  sync(device);c={'signature':signature,'config':asdict(model.cfg),'step':done,'target_steps':steps,'model':model.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':g.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'python_rng':random.getstate(),'real_tokens':real,'supervised_tokens':supervised,'wall_seconds':elapsed+time.perf_counter()-start,'history':history,'best_score':bestscore,'baseline_validation':baseline}
  s4.atomic_torch(last,c);s4.atomic_json(directory/'progress.json',{k:c[k] for k in ['step','target_steps','real_tokens','supervised_tokens','wall_seconds','best_score']})
 if step==0:
  baseline=evaluate(model,None);bestscore=baseline['selection_score'];history.append({'step':0,'validation':baseline});s4.atomic_torch(best,{'model':model.state_dict(),'step':0,'score':bestscore});save(0)
 for i in range(step,steps):
  model.train();opt.zero_grad(set_to_none=True)
  ratio=(i+1)/30 if i<30 else .2+.8*.5*(1+math.cos(math.pi*(i-30)/max(1,steps-30)))
  for pg in opt.param_groups:pg['lr']=lr*ratio
  objective=0.
  for items,n,weight in specifications(i):
   a,b,y,nreal=s4.batch(s4.choose(items,n,g),device)
   with s4.context():value,ntarget=s4.loss(model,a,b,y,.05)
   if not torch.isfinite(value):raise FloatingPointError('Non-finite loss')
   (weight*value/ntarget).backward();objective+=weight*float(value.detach())/ntarget;real+=nreal;supervised+=ntarget
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  if (i+1)%every==0 or i+1==steps:
   ev=evaluate(model,baseline);history.append({'step':i+1,'training_objective':objective,'validation':ev});print('ETAPE5_VALIDATION',i+1,json.dumps(ev),flush=True)
   if ev['eligible'] and ev['selection_score']<bestscore:
    bestscore=ev['selection_score'];s4.atomic_torch(best,{'model':model.state_dict(),'step':i+1,'score':bestscore})
  if (i+1)%save_every==0 or i+1==steps or i+1==interrupt_at:save(i+1)
  if i+1==interrupt_at:return None
 saved=torch.load(best,map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);model.zero_grad(set_to_none=True)
 return {'steps':steps,'best_step':saved['step'],'best_score':bestscore,'history':history,'real_tokens':real,'supervised_tokens':supervised,'wall_seconds':elapsed+time.perf_counter()-start}

def verify_resume():
 previous=s4.DEVICE;s4.DEVICE=torch.device('cpu')
 cfg=S2SConfig(vocab_size=32,dim=48,heads=6,kv_heads=2,encoder_layers=1,decoder_layers=2,ffn_dim=64,dropout=.1)
 items=[{'source':[3,8,9,2],'target':[10,11,2]} for _ in range(5)]
 def specs(i):return [(items,2,1.)]
 def ev(m,b):return {'selection_score':2.,'eligible':True}
 try:
  with tempfile.TemporaryDirectory() as td:
   td=pathlib.Path(td)
   torch.manual_seed(19);random.seed(19);m=MandiaS2S(cfg);train(m,specs,ev,td/'full','test',s4.DEVICE,steps=8,every=4,save_every=4)
   torch.manual_seed(19);random.seed(19);m=MandiaS2S(cfg);train(m,specs,ev,td/'resume','test',s4.DEVICE,steps=8,every=4,save_every=4,interrupt_at=4)
   m=MandiaS2S(cfg);train(m,specs,ev,td/'resume','test',s4.DEVICE,steps=8,every=4,save_every=4)
   a=torch.load(td/'full/checkpoint-last.pt',weights_only=True);b=torch.load(td/'resume/checkpoint-last.pt',weights_only=True)
   for k in a['model']:assert torch.equal(a['model'][k],b['model'][k]),k
   for k,v in a['optimizer']['state'].items():
    for field,tensor in v.items():assert torch.equal(tensor,b['optimizer']['state'][k][field])
   assert torch.equal(a['sampler_rng'],b['sampler_rng']) and torch.equal(a['torch_rng'],b['torch_rng'])
   assert a['real_tokens']==b['real_tokens'] and a['supervised_tokens']==b['supervised_tokens']
  return {'actual_trainer_cpu_resume_bitwise_equal':True,'weights_optimizer_rng_counters':True,'cuda_bitwise_not_claimed':True}
 finally:s4.DEVICE=previous

def main():
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 device=torch.device('cuda');s4.DEVICE=device;torch.set_num_threads(4);ROOT.mkdir(exist_ok=True)
 s4.atomic_json(ROOT/'VERIFICATION-REPRISE.json',verify_resume())
 tok,old=s4.prepare();rows=[json.loads(x) for x in (ROOT/'quotidien-prepare.jsonl').read_text().splitlines()]
 new,excluded=encode(rows,tok);parent=s4.ROOT/'hybrid/model.pt';payload=torch.load(parent,map_location='cpu',weights_only=True)
 assert s4.sha(s4.ROOT/'tokenizer.json')==payload['tokenizer_sha256']
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
 model=MandiaS2S(S2SConfig(**payload['config'])).to(device);model.load_state_dict(payload['state_dict'])
 experiment={'parent_model_sha256':s4.sha(parent),'parent_best_step':1250,'initialization':'own best stage4 weights; fresh AdamW, not stage4 last optimizer','new_data_sha256':s4.sha(ROOT/'quotidien-prepare.jsonl'),'code_sha256':s4.sha(__file__),'tokenizer_sha256':payload['tokenizer_sha256'],'steps':STEPS,'lr':LR,'seed':42,'new_counts':{s:{k:len(v) for k,v in g.items()} for s,g in new.items()},'excluded_length':excluded,'training_mix':'.6 new paired sentences alternating translation/guided reply, .3 old four-domain replay, .1 old 14 reply prompts','selection':'.65 new validation CE + .35 old balanced CE; retain all 14 old replies and old CE <= baseline + .08; no test-based selection','native_validation':False}
 s4.atomic_json(ROOT/'EXPERIMENT.json',experiment);signature=hashlib.sha256(json.dumps(experiment,sort_keys=True).encode()).hexdigest()
 def validate(m,baseline):
  n=new_loss(m,new['validation'],device);o=s4.evaluate(m,old['validation'],device);d,_=demo(m,old,tok,device)
  eligible=d['exact_match']==1. and (baseline is None or o['selection_score']<=baseline['old_validation']+.08)
  return {'new_validation':n,'old_validation':o['selection_score'],'old_reply_exact':d['exact_match'],'eligible':eligible,'selection_score':.65*n['loss']+.35*o['selection_score']}
 def test(m):
  metrics={'new_test':new_loss(m,new['test'],device),'old_test':s4.evaluate(m,old['test'],device),'generation':{},'old_reply':demo(m,old,tok,device)[0]};outputs=[]
  for task,items in new['test'].items():
   metric,out=s4.score_generation(m,sorted(items,key=lambda r:r['row']['id']),tok,device,96);metrics['generation'][task]=metric;outputs.extend(out)
  return metrics,outputs
 beforefile=ROOT/'AVANT.json'
 if not beforefile.exists():
  before,outputs=test(model);s4.atomic_json(beforefile,before);s4.atomic_json(ROOT/'generations-avant.json',outputs)
 else:before=json.loads(beforefile.read_text())
 def specs(i):
  task='translate' if i%2==0 else 'guided';domain=['everyday','software','gospel','lexical'][i%4]
  return [(new['train'][task],12,.6),(old['train'][domain],12,.3),(old['train']['reply'],4,.1)]
 trained=train(model,specs,validate,ROOT,signature,device)
 after,outputs=test(model);s4.atomic_json(ROOT/'generations-apres.json',outputs)
 s4.atomic_torch(ROOT/'model.pt',{'config':payload['config'],'state_dict':model.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256']})
 report={'experiment':experiment,'training':trained,'before':before,'after':after,'new_weights_selected':trained['best_step']>0,'limits':['one seed','community example translations, no Mandia native validation','guided reply means translation request, not free conversation','new corpus has 393 unique source pairs before tokenization, not two independent copies','Bible replay uses admitted stage4 Gospel only; full Bible importer remains disabled until edition/reuse provenance is established']}
 s4.atomic_json(ROOT/'rapport-etape5.json',report)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
 s4.atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',{'verified_on_mounted_drive':True,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':s4.sha(p)} for p in sorted(files)]})
 print('MANDIA_ETAPE5_TERMINE',json.dumps({k:v for k,v in report.items() if k!='training'},ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':main()
