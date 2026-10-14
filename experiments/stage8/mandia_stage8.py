"""Controlled width experiment with grounded Bible QA and daily QA replay.
No external pretrained weights; results only after real Colab execution.
"""
import pathlib,json,hashlib,random,gc,sys
from dataclasses import asdict
import torch
import mandia_stage4 as s4
import mandia_stage5 as s5
import mandia_stage6 as s6
import mandia_stage7 as s7
from mandia_seq2seq import MandiaS2S,S2SConfig
from mandia_expand import widen_ffn
ROOT=s4.ROOT.parent/'mandia-compact-s2s-etape8'
STEPS,LR=400,1e-5

def read(p):return list(map(json.loads,p.read_text().splitlines()))

def main():
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 assert pathlib.Path('/content/drive/MyDrive').exists(),'Drive required before any model load/training'
 device=torch.device('cuda');s4.DEVICE=device;torch.set_num_threads(4);ROOT.mkdir(exist_ok=True)
 tok,old=s4.prepare()
 daily=s7.encode(read(s7.ROOT/'teacher-grounded-qa.jsonl'),tok)
 bible=s7.encode(read(ROOT/'bible-qa-pilote.jsonl'),tok)
 prior5,_=s5.encode(read(s5.ROOT/'quotidien-prepare.jsonl'),tok)
 prior6,_=s6.encode(read(s6.ROOT/'dialogues-prepare.jsonl'),tok)
 parent=s7.ROOT/'teacher_french/model.pt'
 assert s4.sha(parent)=='e6fe5b71079842d2644dba26ccfcc03c59fdfd0a9e4cdb509afad1ff613b885e'
 payload=torch.load(parent,map_location='cpu',weights_only=True)
 assert payload['tokenizer_sha256']==s4.sha(s4.ROOT/'tokenizer.json')
 experiment={'stage':8,'parent_sha256':s4.sha(parent),'tokenizer_sha256':payload['tokenizer_sha256'],
  'steps_per_arm':STEPS,'lr':LR,'seed':42,'warmup':40,'optimizer':'fresh AdamW per arm, beta(.9,.98), wd.01, smooth.05, clip1',
  'arms':{'small_13m':768,'wide_20m':1536},'scope':'pilot width ablation, same data and sampled tokens; compute NOT matched',
  'new_data':'64 views, 40 target families, 38 existing Gospel source rows, inherited chapter split; no new real daily dialogues',
  'literature':['https://aclanthology.org/2024.findings-acl.30/','https://arxiv.org/abs/1511.05641','https://aclanthology.org/2025.lowresnlp-1.6/'],
  'mix':'.25 BibleQA8, .35 dailyQA16, .15 oldSMOLreply8, .15 balancedold8, .05 dailytranslation8, .05 oldreply4',
  'selection':'.35 Bible validation CE + .35 dailyQA validation CE + .2 old balanced CE + .1 SMOL validation CE; retain14reply; oldCE<=initial+.05, SMOLCE<=initial+.1, dailyQACE<=initial+.15; freeze winner before test',
  'native_validation':False,'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
  'data_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in [s4.ROOT/'parallel-prepare.jsonl',s4.ROOT/'replies-prepare.jsonl',s4.ROOT/'data/waxal-baoule-text.jsonl',s5.ROOT/'quotidien-prepare.jsonl',s6.ROOT/'dialogues-prepare.jsonl',s7.ROOT/'teacher-grounded-qa.jsonl',ROOT/'bible-qa-pilote.jsonl']},
  'code_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in [pathlib.Path(__file__),ROOT/'mandia_expand.py',pathlib.Path(s7.__file__)]+list(pathlib.Path(s6.__file__).parent.glob('*.py'))}}
 s4.atomic_json(ROOT/'EXPERIMENT.json',experiment)
 def parent_model():
  m=MandiaS2S(S2SConfig(**payload['config']));m.load_state_dict(payload['state_dict']);return m
 # Full trained model check, separate from the random CPU fixture.
 small=parent_model().to(device).eval();wide=widen_ffn(small).to(device).eval()
 a,b,y,_=s4.batch(bible['train'][:4],device)
 with torch.inference_mode():
  z=small(a,b);w=wide(a,b);delta=float((z-w).abs().max());torch.testing.assert_close(z,w,atol=5e-5,rtol=5e-4)
  with s4.context():zb=small(a,b);wb=wide(a,b)
  check={'trained_weights_fp32_max_absolute_error':delta,'fp32_close':True,'bf16_max_absolute_error':float((zb-wb).abs().max()),'bf16_argmax_fraction_equal':float((zb.argmax(-1)==wb.argmax(-1)).float().mean())}
 # Gradient checks must run outside inference_mode.
 check['cpu_fixture']=__import__('mandia_expand').verify()
 torch.set_num_threads(4)
 s4.atomic_json(ROOT/'VERIFICATION-ELARGISSEMENT.json',check)
 del small,wide,z,w,zb,wb;gc.collect();torch.cuda.empty_cache()
 oldlr=s6.LR;s6.LR=LR
 try:s4.atomic_json(ROOT/'VERIFICATION-REPRISE.json',s6.verify())
 finally:s6.LR=oldlr
 def validate(m,base):
  q=s7.qa_loss(m,bible['validation'],device)['loss'];d=s7.qa_loss(m,daily['validation'],device)['loss'];o=s4.evaluate(m,old['validation'],device)['selection_score'];p=s6.losses(m,prior6['validation'],device)['loss'];r=s5.demo(m,old,tok,device)[0]['exact_match']
  return {'bible_ce':q,'daily_ce':d,'old_ce':o,'smol_ce':p,'reply14_exact':r,'selection_score':.35*q+.35*d+.2*o+.1*p,'eligible':r==1 and (base is None or (o<=base['old_ce']+.05 and p<=base['smol_ce']+.1 and d<=base['daily_ce']+.15))}
 def specs(i):
  domain=['everyday','software','gospel','lexical'][i%4]
  return [(bible['train'],8,.25,False),(daily['train'],16,.35,False),(prior6['train']['reply'],8,.15,False),(old['train'][domain],8,.15,False),(prior5['train']['translate'],8,.05,False),(old['train']['reply'],4,.05,False)]
 arms={}
 for name,ffn in experiment['arms'].items():
  m=parent_model()
  if ffn!=768:m=widen_ffn(m,ffn)
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
  m=m.to(device);cfg=asdict(m.cfg);signature=hashlib.sha256(json.dumps({'experiment':experiment,'arm':name},sort_keys=True).encode()).hexdigest()
  oldlr=s6.LR;s6.LR=LR
  try:trained=s6.train(m,specs,validate,ROOT/name,signature,device,0.,steps=STEPS,every=50)
  finally:s6.LR=oldlr
  s4.atomic_torch(ROOT/name/'model.pt',{'config':cfg,'state_dict':m.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256']})
  arms[name]={'training':trained,'parameters':sum(p.numel() for p in m.parameters())}
  s4.atomic_json(ROOT/name/'train-report.json',arms[name]);del m;gc.collect();torch.cuda.empty_cache()
 assert len({v['training']['real_tokens'] for v in arms.values()})==1
 assert len({v['training']['supervised_tokens'] for v in arms.values()})==1
 chosen=min(arms,key=lambda k:arms[k]['training']['best_score'])
 s4.atomic_json(ROOT/'SELECTION-AVANT-TEST.json',{'winner':chosen,'scores':{k:v['training']['best_score'] for k,v in arms.items()},'test_not_used':True})
 def test(m,label):
  result={};gens={}
  for key,items in [('bible',bible['test']),('daily',daily['test'])]:
   result[key]={'ce':s7.qa_loss(m,items,device)['loss']};result[key]['generation'],gens[key]=s4.score_generation(m,items,tok,device,96)
  result['old_stage4']=s4.evaluate(m,old['test'],device);result['old_stage5']=s5.new_loss(m,prior5['test'],device);result['old_stage6']=s6.losses(m,prior6['test'],device);result['trained14']=s5.demo(m,old,tok,device)[0]
  s4.atomic_json(ROOT/f'generations-{label}.json',gens);return result
 m=parent_model().to(device);before=test(m,'parent');del m;gc.collect();torch.cuda.empty_cache()
 for name in arms:
  p=torch.load(ROOT/name/'model.pt',map_location='cpu',weights_only=True);m=MandiaS2S(S2SConfig(**p['config'])).to(device);m.load_state_dict(p['state_dict']);del p
  arms[name]['test']=test(m,name);del m;gc.collect();torch.cuda.empty_cache()
 report={'experiment':experiment,'before':before,'arms':arms,'selected_by_validation':chosen,'same_real_and_supervised_tokens_verified':True,'limits':['tiny Bible QA pilot; no new natural daily conversations','one seed, not equal FLOPs','Bible chapter holdouts have been evaluated previously for translation; stories can overlap','native naturalness not assessed; no fluency or efficiency superiority claim']}
 s4.atomic_json(ROOT/'rapport-etape8.json',report)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
 s4.atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',{'verified_on_mounted_drive':True,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':s4.sha(p)} for p in sorted(files)]})
 print('MANDIA_ETAPE8_TERMINE',json.dumps({'chosen':chosen,'before':before['bible'],'arms':{k:{'parameters':v['parameters'],'bible':v['test']['bible'],'daily':v['test']['daily']} for k,v in arms.items()}}),flush=True)
 return report

if __name__=='__main__':main()
