"""Expanded source-grounded curriculum versus the earlier Bible pilot.
Own 20.86M parent, unchanged architecture, no generated Baoule labels.
"""
import pathlib,json,hashlib,random,gc
import torch
import mandia_stage4 as s4
import mandia_stage5 as s5
import mandia_stage6 as s6
import mandia_stage7 as s7
from mandia_seq2seq import MandiaS2S,S2SConfig
ROOT=s4.ROOT.parent/'mandia-compact-s2s-etape9'
STEPS,LR=600,1e-5
def read(p):return list(map(json.loads,p.read_text().splitlines()))
def main():
 assert pathlib.Path('/content/drive/MyDrive').exists(),'Drive must be mounted before training'
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 device=torch.device('cuda');s4.DEVICE=device;torch.set_num_threads(4);ROOT.mkdir(exist_ok=True)
 tok,old=s4.prepare()
 daily=s7.encode(read(s7.ROOT/'teacher-grounded-qa.jsonl'),tok)
 broad=s7.encode(read(ROOT/'quotidien-contextuel.jsonl'),tok)
 bible=s7.encode(read(ROOT/'bible-etendue.jsonl'),tok)
 pilot=s7.encode(read(ROOT.parent/'mandia-compact-s2s-etape8/bible-qa-pilote.jsonl'),tok)
 written=s7.encode(read(ROOT/'dialogues-ecrits-ajout.jsonl'),tok)
 p5,_=s5.encode(read(s5.ROOT/'quotidien-prepare.jsonl'),tok)
 p6,_=s6.encode(read(s6.ROOT/'dialogues-prepare.jsonl'),tok)
 parent=ROOT.parent/'mandia-compact-s2s-etape8/wide_20m/model.pt'
 assert s4.sha(parent)=='6f2bac99307900f96d80a6b9d4980a3b5bc19fc07486beff456979b2d3decbee'
 payload=torch.load(parent,map_location='cpu',weights_only=True)
 assert payload['tokenizer_sha256']==s4.sha(s4.ROOT/'tokenizer.json')
 def load_parent():
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
  m=MandiaS2S(S2SConfig(**payload['config'])).to(device);m.load_state_dict(payload['state_dict']);return m
 paths=[ROOT/n for n in ['bible-etendue.jsonl','quotidien-contextuel.jsonl','dialogues-ecrits-ajout.jsonl']]+[s4.ROOT/'parallel-prepare.jsonl',s4.ROOT/'replies-prepare.jsonl',s5.ROOT/'quotidien-prepare.jsonl',s6.ROOT/'dialogues-prepare.jsonl',s7.ROOT/'teacher-grounded-qa.jsonl',ROOT.parent/'mandia-compact-s2s-etape8/bible-qa-pilote.jsonl']
 experiment={'stage':9,'date':'2026-10-14','parent_sha256':s4.sha(parent),'tokenizer_sha256':payload['tokenizer_sha256'],
  'architecture':'unchanged own 20,860,416 parameters; no external pretrained model','steps_per_arm':STEPS,'lr':LR,'seed':42,
  'optimizer':'fresh AdamW(.9,.98), wd.01, label smooth.05, clip1, warm40 then cosine to20%',
  'arms':['pilot_control','expanded_bible'],'difference':'Bible TRAIN group only: earlier pilot vs expanded curriculum. Both arms receive identical new daily and SMOL data, group weights and batch counts; token counts and FLOPs differ.',
  'mix':'.25 known daily QA16, .15 broad contextual daily8, .20 Bible8, .10 new written SMOL4, .15 old SMOL reply8, .10 old domain8, .05 old reply4',
  'selection':'.30 known daily validation CE + .15 broad daily CE + .20 expanded Bible CE + .10 new written SMOL CE + .15 old SMOL CE + .10 balanced old CE; retain14, oldCE<=initial+.05, oldSMOL<=initial+.10, knownDaily<=initial+.10',
  'teacher_role':'French only; source Baoule preserved. No assistant teacher logits, no self-training on unchecked Baoule.',
  'native_validation':False,'data_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in paths},
  'code_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in [pathlib.Path(__file__),pathlib.Path(s7.__file__)]+list(pathlib.Path(s6.__file__).parent.glob('*.py'))},
  'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
  'scope':'single seed curriculum ablation; known40 is observed regression, fresh SMOL test has only one document'}
 s4.atomic_json(ROOT/'EXPERIMENT.json',experiment)
 def validate(m,base):
  d=s7.qa_loss(m,daily['validation'],device)['loss'];w=s7.qa_loss(m,broad['validation'],device)['loss']
  b=s7.qa_loss(m,bible['validation'],device)['loss'];n=s7.qa_loss(m,written['validation'],device)['loss']
  o=s4.evaluate(m,old['validation'],device)['selection_score'];p=s6.losses(m,p6['validation'],device)['loss'];r=s5.demo(m,old,tok,device)[0]['exact_match']
  return {'known_daily_ce':d,'broad_daily_ce':w,'bible_ce':b,'new_smol_ce':n,'old_smol_ce':p,'old_ce':o,'trained14':r,
   'selection_score':.30*d+.15*w+.20*b+.10*n+.15*p+.10*o,
   'eligible':r==1 and (base is None or (o<=base['old_ce']+.05 and p<=base['old_smol_ce']+.10 and d<=base['known_daily_ce']+.10))}
 oldlr=s6.LR;s6.LR=LR
 try:s4.atomic_json(ROOT/'VERIFICATION-REPRISE.json',s6.verify())
 finally:s6.LR=oldlr
 arms={}
 for name,group in [('pilot_control',pilot),('expanded_bible',bible)]:
  def specs(i):
   domain=['everyday','software','gospel','lexical'][i%4]
   return [(daily['train'],16,.25,False),(broad['train'],8,.15,False),(group['train'],8,.20,False),(written['train'],4,.10,False),(p6['train']['reply'],8,.15,False),(old['train'][domain],8,.10,False),(old['train']['reply'],4,.05,False)]
  m=load_parent();signature=hashlib.sha256(json.dumps({'experiment':experiment,'arm':name},sort_keys=True).encode()).hexdigest()
  oldlr=s6.LR;s6.LR=LR
  try:trained=s6.train(m,specs,validate,ROOT/name,signature,device,0.,steps=STEPS,every=50)
  finally:s6.LR=oldlr
  s4.atomic_torch(ROOT/name/'model.pt',{'config':payload['config'],'state_dict':m.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256']})
  arms[name]={'training':trained,'parameters':sum(p.numel() for p in m.parameters())};s4.atomic_json(ROOT/name/'train-report.json',arms[name])
  del m;gc.collect();torch.cuda.empty_cache()
 winner=min(arms,key=lambda k:arms[k]['training']['best_score'])
 s4.atomic_json(ROOT/'SELECTION-AVANT-TEST.json',{'winner':winner,'scores':{k:v['training']['best_score'] for k,v in arms.items()},'test_not_used':True})
 # Fixed deterministic samples defined independently of model outputs. Original
 # translation holdouts have been observed previously; these are new formats.
 def subset(items,n):return sorted(items,key=lambda x:hashlib.sha256(x['row']['id'].encode()).hexdigest())[:n]
 testgroups={'known_daily40':daily['test'],'prior_bible8':pilot['test'],'expanded_bible12':subset(bible['test'],12),'broad_daily16':subset(broad['test'],16),'new_written_dialogue5':written['test']}
 def test(m,label):
  result={};gens={}
  for key,items in testgroups.items():
   result[key]={'ce':s7.qa_loss(m,items,device)['loss']};result[key]['generation'],gens[key]=s4.score_generation(m,items,tok,device,128 if key in ['expanded_bible12','new_written_dialogue5'] else 96)
  result['old_stage4']=s4.evaluate(m,old['test'],device);result['old_stage5']=s5.new_loss(m,p5['test'],device);result['old_stage6']=s6.losses(m,p6['test'],device);result['trained14']=s5.demo(m,old,tok,device)[0]
  s4.atomic_json(ROOT/f'generations-{label}.json',gens);return result
 m=load_parent();before=test(m,'parent');del m;gc.collect();torch.cuda.empty_cache()
 for name in arms:
  p=torch.load(ROOT/name/'model.pt',map_location='cpu',weights_only=True);m=load_parent();m.load_state_dict(p['state_dict']);del p
  arms[name]['test']=test(m,name);del m;gc.collect();torch.cuda.empty_cache()
 report={'experiment':experiment,'before':before,'arms':arms,'selected_by_validation':winner,
  'limits':['known40 is a previously observed regression with known target families, not conversational accuracy',
  'new written dialogue test: 5 replies from one source document, no native validation',
  'Bible and Wiktionary translation holdouts were already evaluated in earlier stages; QA formats differ',
  'expanded Bible mainly reformulation views; no full Bible imported and no Louis Segond lineage assumed',
  'same architecture and number of updates, not matched tokens/FLOPs; one seed only',
  'new data and continued training are combined in parent comparison; Bible-specific effect assessed against pilot_control only']}
 s4.atomic_json(ROOT/'rapport-etape9.json',report)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
 s4.atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',{'verified_on_mounted_drive':True,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':s4.sha(p)} for p in sorted(files)]})
 print('MANDIA_ETAPE9_TERMINE',json.dumps({'chosen':winner,'before':before['known_daily40'],'arms':{k:{'best_step':v['training']['best_step'],'daily':v['test']['known_daily40'],'new_dialogue':v['test']['new_written_dialogue5']} for k,v in arms.items()}},ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':main()
