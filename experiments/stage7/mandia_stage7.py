"""Teacher-assisted French augmentation, fixed own 13.78M model.
Known-target phrasing split, NOT unseen-target or native conversational evaluation.
"""
import pathlib,json,hashlib,random,gc,copy
import torch
import mandia_stage4 as s4
import mandia_stage5 as s5
import mandia_stage6 as s6
from mandia_seq2seq import MandiaS2S,S2SConfig
ROOT=s4.ROOT.parent/'mandia-compact-s2s-etape7'
STEPS,LR=400,2e-5

def encode(rows,tok,canonical=False):
 result={s:[] for s in ['train','validation','test']}
 for r in rows:
  prompt=r['canonical_prompt'] if canonical and r['split']=='train' else r['prompt']
  item={'source':[4]+tok.encode(prompt).ids+[2],'target':tok.encode(r['answer']).ids+[2],'row':dict(r,prompt=prompt)}
  assert len(item['source'])<=160 and len(item['target'])<=192,r['id']
  result[r['split']].append(item)
 return result

@torch.inference_mode()
def qa_loss(model,items,device):
 model.eval();total=tokens=0
 for start in range(0,len(items),16):
  a,b,y,_=s4.batch(items[start:start+16],device)
  with s4.context():v,n=s4.loss(model,a,b,y)
  total+=float(v);tokens+=n
 return {'loss':total/tokens,'rows':len(items),'target_tokens':tokens}

def verify_paired_sampling(control,teacher):
 a=torch.Generator().manual_seed(42);b=torch.Generator().manual_seed(42)
 for i in range(STEPS):
  x=s4.choose(control,16,a);y=s4.choose(teacher,16,b)
  assert [r['row']['id'] for r in x]==[r['row']['id'] for r in y]
  assert [r['target'] for r in x]==[r['target'] for r in y]
 return {'same_drawn_new_row_ids_and_targets':True,'input_lengths_may_differ':True,'sampling_only_fixture':True}

def main():
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 device=torch.device('cuda');s4.DEVICE=device;torch.set_num_threads(4);ROOT.mkdir(exist_ok=True)
 tok,old=s4.prepare()
 prior5,_=s5.encode([json.loads(l) for l in (s5.ROOT/'quotidien-prepare.jsonl').read_text().splitlines()],tok)
 prior6,_=s6.encode([json.loads(l) for l in (s6.ROOT/'dialogues-prepare.jsonl').read_text().splitlines()],tok)
 rows=[json.loads(l) for l in (ROOT/'teacher-grounded-qa.jsonl').read_text().splitlines()]
 teacher=encode(rows,tok);control=encode(rows,tok,True)
 assert all(a['target']==b['target'] for a,b in zip(control['train'],teacher['train']))
 parent=s6.ROOT/'ce_control/model.pt'
 assert s4.sha(parent)=='a2afe5128d6ba5f807e0dc28baffdf606708bde96610469837d9d18cea8cb2ad'
 payload=torch.load(parent,map_location='cpu',weights_only=True)
 assert payload['tokenizer_sha256']==s4.sha(s4.ROOT/'tokenizer.json')
 all_data=[s4.ROOT/'parallel-prepare.jsonl',s4.ROOT/'replies-prepare.jsonl',s4.ROOT/'data/waxal-baoule-text.jsonl',s5.ROOT/'quotidien-prepare.jsonl',s6.ROOT/'dialogues-prepare.jsonl',ROOT/'teacher-grounded-qa.jsonl']
 experiment={'stage':7,'parent_model_sha256':s4.sha(parent),'tokenizer_sha256':payload['tokenizer_sha256'],'data_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in all_data},'code_sha256':{p.name:s4.sha(p) for p in list(map(pathlib.Path,[__file__]))+list(pathlib.Path(s6.__file__).parent.glob('*.py'))},'steps_per_arm':STEPS,'lr':LR,'seed':42,'warmup':40,'architecture':'unchanged own 13,782,528 parameters','initialization':'selected stage6 CE step100, fresh AdamW per arm','contrastive_weight':0,'teacher_role':'French contextual questions/paraphrases only; original Baoule exact source examples; no teacher logits/weights/rationales','counts':{s:len(v) for s,v in teacher.items()},'target_families':40,'split_scope':'held-out French views of SAME KNOWN target families, all stage5 TRAIN; not unseen-target evaluation','mix':'.5 contextual QA16, .2 old SMOL reply8, .15 old domain8, .1 stage5 translation8, .05 old reply4','selection':'.6 QA validation CE + .2 stage4 balanced CE + .2 stage6 validation CE; retain14oldreplies; oldCE<=initial+.05; stage6CE<=initial+.10; winner frozen before test','matched_budget':'same sampled identities and supervised tokens; actual source tokens differ and are measured, not a matched compute architecture benchmark','native_validation':False,'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}
 s4.atomic_json(ROOT/'EXPERIMENT.json',experiment)
 s4.atomic_json(ROOT/'VERIFICATION-SAMPLING.json',verify_paired_sampling(control['train'],teacher['train']))
 # Reuse the previously verified exact-resume trainer without editing its frozen source.
 old_lr=s6.LR;s6.LR=LR
 try:s4.atomic_json(ROOT/'VERIFICATION-REPRISE.json',s6.verify())
 finally:s6.LR=old_lr
 def validate(m,base):
  q=qa_loss(m,teacher['validation'],device);o=s4.evaluate(m,old['validation'],device)['selection_score'];p=s6.losses(m,prior6['validation'],device)['loss'];d=s5.demo(m,old,tok,device)[0]['exact_match']
  return {'qa_validation':q,'old_validation':o,'stage6_validation':p,'old_reply_exact':d,'eligible':d==1 and (base is None or (o<=base['old_validation']+.05 and p<=base['stage6_validation']+.10)),'selection_score':.6*q['loss']+.2*o+.2*p}
 def load_parent():
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
  m=MandiaS2S(S2SConfig(**payload['config'])).to(device);m.load_state_dict(payload['state_dict']);return m
 arms={}
 for name,groups in [('canonical_control',control),('teacher_french',teacher)]:
  def specs(i):
   domain=['everyday','software','gospel','lexical'][i%4]
   return [(groups['train'],16,.5,False),(prior6['train']['reply'],8,.2,False),(old['train'][domain],8,.15,False),(prior5['train']['translate'],8,.1,False),(old['train']['reply'],4,.05,False)]
  m=load_parent();signature=hashlib.sha256(json.dumps({'experiment':experiment,'arm':name},sort_keys=True).encode()).hexdigest()
  old_lr=s6.LR;s6.LR=LR
  try:trained=s6.train(m,specs,validate,ROOT/name,signature,device,0.,steps=STEPS,every=50)
  finally:s6.LR=old_lr
  s4.atomic_torch(ROOT/name/'model.pt',{'config':payload['config'],'state_dict':m.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256']})
  arms[name]={'training':trained};s4.atomic_json(ROOT/name/'train-report.json',arms[name])
  del m;gc.collect();torch.cuda.empty_cache()
 assert len({v['training']['supervised_tokens'] for v in arms.values()})==1
 chosen=min(arms,key=lambda name:arms[name]['training']['best_score'])
 s4.atomic_json(ROOT/'SELECTION-AVANT-TEST.json',{'winner':chosen,'scores':{k:v['training']['best_score'] for k,v in arms.items()},'test_not_used':True})
 def test(m):
  result={'qa':qa_loss(m,teacher['test'],device),'old_stage4':s4.evaluate(m,old['test'],device),'old_stage5':s5.new_loss(m,prior5['test'],device),'old_stage6':s6.losses(m,prior6['test'],device),'trained14':s5.demo(m,old,tok,device)[0]}
  result['generation'],generations=s4.score_generation(m,teacher['test'],tok,device,96)
  # Identical questions with different explicit contexts: two small, known-target checks.
  index={g['id']:g for g in generations};pairs=[]
  for x,y in [(112,249),(117,188)]:
   a=index[f'teacher-{x:04d}-4'];b=index[f'teacher-{y:04d}-4']
   assert a['prompt_fr'].split('\nMessage : ')[1]==b['prompt_fr'].split('\nMessage : ')[1]
   pairs.append({'ids':[a['id'],b['id']],'identical_question':True,'outputs_differ':s4.canonical(a['generated_bci'])!=s4.canonical(b['generated_bci']),'both_exact':all(s4.canonical(g['generated_bci'])==s4.canonical(g['reference_bci']) for g in [a,b])})
  result['two_context_pairs']=pairs
  return result,generations
 m=load_parent();before,gens=test(m);s4.atomic_json(ROOT/'generations-avant.json',gens);del m;gc.collect();torch.cuda.empty_cache()
 for name in arms:
  m=load_parent();c=torch.load(ROOT/name/'model.pt',map_location='cpu',weights_only=True);m.load_state_dict(c['state_dict']);del c
  arms[name]['test'],gens=test(m);s4.atomic_json(ROOT/name/'generations-test.json',gens);del m;gc.collect();torch.cuda.empty_cache()
 report={'experiment':experiment,'before':before,'arms':arms,'selected_by_validation':chosen,'same_supervised_tokens_verified':True,'parameters':13782528,'limits':['40 source target families already seen in parent translation training','only French wording and contextual question format held out; no unseen-target claim','French authored by assistant, no authentic recorded new conversations, no native validation','one seed; no teacher logits or weights; this is teacher-assisted augmentation','input tokens differ between arms; not a matched compute or architecture speed benchmark','old tests previously observed; regression checks only']}
 s4.atomic_json(ROOT/'rapport-etape7.json',report)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
 s4.atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',{'verified_on_mounted_drive':True,'selected_by_validation':chosen,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':s4.sha(p)} for p in sorted(files)]})
 print('MANDIA_ETAPE7_TERMINE',json.dumps({'winner':chosen,'before':before['generation'],'after':{k:v['test']['generation'] for k,v in arms.items()}},ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':main()
