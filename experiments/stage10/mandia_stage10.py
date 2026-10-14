"""French-only translation, longer aligned text and held-out source families.
Own Mandia 20.86M, context ceiling 512, no new learned parameters.
"""
import pathlib,json,hashlib,random,gc
import torch
import mandia_stage4 as s4
import mandia_stage5 as s5
import mandia_stage6 as s6
import mandia_stage7 as s7
from mandia_seq2seq import MandiaS2S,S2SConfig
ROOT=s4.ROOT.parent/'mandia-compact-s2s-etape10'
STEPS,LR=1200,2e-5
def read(p):return list(map(json.loads,p.read_text().splitlines()))
def encode(rows,tok):
 out={s:{'sentences':[],'long':[],'reply':[]} for s in ['train','validation','test']}
 for r in rows:
  task='reply' if r['task']=='reply' else 'long' if r['domain'] in ['new_paragraph_translation','whole_document_translation'] else 'sentences'
  item={'source':[4 if task=='reply' else 3]+tok.encode(r['prompt']).ids+[2],'target':tok.encode(r['answer']).ids+[2],'row':r}
  assert len(item['source'])<=512 and len(item['target'])<=512
  out[r['split']][task].append(item)
 assert all(v for g in out.values() for v in g.values())
 return out
@torch.inference_mode()
def ce(m,items,device):
 m.eval();total=count=0
 for i in range(0,len(items),4):
  a,b,y,_=s4.batch(items[i:i+4],device)
  with s4.context():v,n=s4.loss(m,a,b,y)
  total+=float(v);count+=n
 return total/count
def verify_context():
 previous=s4.DEVICE;s4.DEVICE=torch.device('cpu');torch.manual_seed(17)
 try:
  cfg=S2SConfig(vocab_size=32,dim=48,heads=6,kv_heads=2,encoder_layers=1,decoder_layers=3,ffn_dim=64,context=512,dropout=0)
  m=MandiaS2S(cfg).eval();a=torch.randint(5,32,(1,37));b=torch.randint(5,32,(1,281))
  with torch.inference_mode():
   full=m(a,b);memory,valid=m.encode(a)
   _,state=m.decode(b[:,:280],memory,valid,cache=True)
   cached,_=m.decode(b[:,280:],memory,valid,state,280,True)
   torch.testing.assert_close(full[:,-1:],cached,atol=1e-5,rtol=1e-4)
  return {'tiny_cpu_fixture_beyond_old_256_cap':True,'full_vs_cached_last_position':280,'max_error':float((full[:,-1:]-cached).abs().max()),'no_new_learned_parameters':True,'scope':'cache correctness, not evidence of linguistic long-context ability'}
 finally:s4.DEVICE=previous
def main():
 assert pathlib.Path('/content/drive/MyDrive').exists(),'Mounted Drive required before loading/training'
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 device=torch.device('cuda');s4.DEVICE=device;torch.set_num_threads(4);ROOT.mkdir(exist_ok=True)
 tok,old=s4.prepare();new=encode(read(ROOT/'nouveaux-paralleles.jsonl'),tok)
 p5,_=s5.encode(read(s5.ROOT/'quotidien-prepare.jsonl'),tok);p6,_=s6.encode(read(s6.ROOT/'dialogues-prepare.jsonl'),tok)
 daily=s7.encode(read(s7.ROOT/'teacher-grounded-qa.jsonl'),tok)
 r9=s4.ROOT.parent/'mandia-compact-s2s-etape9'
 bible=s7.encode(read(r9/'bible-etendue.jsonl'),tok);written9=s7.encode(read(r9/'dialogues-ecrits-ajout.jsonl'),tok)
 replies=new['train']['reply']+written9['train']
 parent=r9/'expanded_bible/model.pt'
 assert s4.sha(parent)=='f9932b33108013ebd0891295b8820d82cc98176e555d755eed0a83ddbcf4d0b9'
 payload=torch.load(parent,map_location='cpu',weights_only=True);assert payload['tokenizer_sha256']==s4.sha(s4.ROOT/'tokenizer.json')
 cfg=dict(payload['config'],context=512)
 def load_parent():
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
  m=MandiaS2S(S2SConfig(**cfg)).to(device);m.load_state_dict(payload['state_dict']);return m
 s4.atomic_json(ROOT/'VERIFICATION-CONTEXTE.json',verify_context());torch.set_num_threads(4)
 oldlr=s6.LR;s6.LR=LR
 try:s4.atomic_json(ROOT/'VERIFICATION-REPRISE.json',s6.verify())
 finally:s6.LR=oldlr
 paths=[ROOT/'nouveaux-paralleles.jsonl',s4.ROOT/'parallel-prepare.jsonl',s5.ROOT/'quotidien-prepare.jsonl',s6.ROOT/'dialogues-prepare.jsonl',s7.ROOT/'teacher-grounded-qa.jsonl',r9/'bible-etendue.jsonl',r9/'dialogues-ecrits-ajout.jsonl']
 experiment={'stage':10,'date':'2026-10-14','parent_sha256':s4.sha(parent),'tokenizer_sha256':payload['tokenizer_sha256'],'steps':STEPS,'lr':LR,'seed':42,
  'architecture':'unchanged own 20,860,416 learned parameters; context ceiling 256->512 only; same RoPE and weights',
  'reference':'same parent stage9 weights with context ceiling512, without training, so both can be evaluated on long inputs',
  'new_data_counts':{s:{k:len(v) for k,v in g.items()} for s,g in new.items()},
  'mix':'.25 new sentence translation8, .20 new paragraphs/documents2, .15 new+stage9 written replies4, .20 stage6 translation/reply8 alternating, .10 balanced old8, .05 wiki translation8, .03 prior BibleQA4, .02 old reply4',
  'selection':'.45 new sentence validation CE + .20 new paragraph/document CE + .20 old SMOL reply validation CE + .10 authored birthday-advice QA CE + .05 old balanced CE; retain14; oldCE<=initial+.15; oldSMOLreply<=initial+.10; no known40 in selection',
  'fresh_test':'12 new sentence families, 2 new document families for translation, 2 new written-dialogue families; no source families shared across new splits; test after validation selection fixed',
  'limits':['one seed and one continuation, no architecture or curriculum ablation','new French written by assistant from source English; no native Baoule validation','new dialogues written, not recorded; only5 fresh next-turn test replies','birthday reply validation uses authored questions, not authentic conversational turns','long views reuse sentence content within same source partition','512 token ceiling enables execution, not reliable semantic long-text translation'],
  'data_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in paths},
  'code_sha256':{str(p.relative_to(ROOT.parent)):s4.sha(p) for p in [pathlib.Path(__file__),pathlib.Path(s7.__file__)]+list(pathlib.Path(s6.__file__).parent.glob('*.py'))},'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}
 s4.atomic_json(ROOT/'EXPERIMENT.json',experiment)
 def validate(m,base):
  n=ce(m,new['validation']['sentences'],device);l=ce(m,new['validation']['long'],device);q=ce(m,new['validation']['reply'],device)
  p=ce(m,p6['validation']['reply'],device);o=s4.evaluate(m,old['validation'],device)['selection_score'];r=s5.demo(m,old,tok,device)[0]['exact_match']
  return {'new_sentence_ce':n,'new_long_ce':l,'new_authored_reply_ce':q,'old_smol_reply_ce':p,'old_ce':o,'trained14':r,'selection_score':.45*n+.20*l+.20*p+.10*q+.05*o,
   'eligible':r==1 and (base is None or (o<=base['old_ce']+.15 and p<=base['old_smol_reply_ce']+.10))}
 def specs(i):
  domain=['everyday','software','gospel','lexical'][i%4];task='translate' if i%2==0 else 'reply'
  return [(new['train']['sentences'],8,.25,False),(new['train']['long'],2,.20,False),(replies,4,.15,False),(p6['train'][task],8,.20,False),(old['train'][domain],8,.10,False),(p5['train']['translate'],8,.05,False),(bible['train'],4,.03,False),(old['train']['reply'],4,.02,False)]
 m=load_parent();assert sum(p.numel() for p in m.parameters())==20860416
 # Same short-input outputs after raising the hard limit, actual parent weights.
 original=MandiaS2S(S2SConfig(**payload['config'])).to(device).eval();original.load_state_dict(payload['state_dict']);m.eval()
 a,b,y,_=s4.batch(new['train']['sentences'][:2],device)
 with torch.inference_mode():
  z=original(a,b);w=m(a,b);assert torch.equal(z,w)
 s4.atomic_json(ROOT/'VERIFICATION-PARENT.json',{'same_short_logits_bitwise_fp32':True,'original_context':256,'new_context':512,'parameters':20860416})
 del original,z,w;gc.collect();torch.cuda.empty_cache()
 signature=hashlib.sha256(json.dumps(experiment,sort_keys=True).encode()).hexdigest();oldlr=s6.LR;s6.LR=LR
 try:trained=s6.train(m,specs,validate,ROOT/'translation_dialogue',signature,device,0.,steps=STEPS,every=100)
 finally:s6.LR=oldlr
 s4.atomic_torch(ROOT/'translation_dialogue/model.pt',{'config':cfg,'state_dict':m.state_dict(),'tokenizer_sha256':payload['tokenizer_sha256']})
 s4.atomic_json(ROOT/'train-report.json',trained)
 s4.atomic_json(ROOT/'SELECTION-AVANT-TEST.json',{'selected_step':trained['best_step'],'validation_score':trained['best_score'],'known40_not_used':True,'new_test_not_used':True})
 del m;gc.collect();torch.cuda.empty_cache()
 def test(model,label):
  groups={'fresh_sentence_translation12':[x for x in new['test']['sentences'] if x['row']['domain']=='new_sentence_translation'],
   'fresh_document_sentences26':[x for x in new['test']['sentences'] if x['row']['domain']=='new_document_sentence'],
   'fresh_paragraphs7':[x for x in new['test']['long'] if x['row']['domain']=='new_paragraph_translation'],
   'fresh_whole_documents2':[x for x in new['test']['long'] if x['row']['domain']=='whole_document_translation'],
   'fresh_written_replies5':new['test']['reply'],'previous_written_replies5':written9['test'],'known_daily40':daily['test']}
  metrics={};outputs={}
  for k,items in groups.items():
   cap=512 if k.startswith('fresh_') else 192 if k.startswith('previous_') else 96
   metrics[k]={'ce':ce(model,items,device)};metrics[k]['generation'],outputs[k]=s4.score_generation(model,items,tok,device,cap)
   metrics[k]['generated_tokens']=[len(tok.encode(r['generated_bci']).ids) for r in outputs[k]]
  metrics['old_stage4']=s4.evaluate(model,old['test'],device);metrics['old_stage5']=s5.new_loss(model,p5['test'],device);metrics['old_stage6']=s6.losses(model,p6['test'],device);metrics['trained14']=s5.demo(model,old,tok,device)[0]
  s4.atomic_json(ROOT/f'generations-{label}.json',outputs);return metrics
 before_model=load_parent();before=test(before_model,'parent_context512');del before_model;gc.collect();torch.cuda.empty_cache()
 p=torch.load(ROOT/'translation_dialogue/model.pt',map_location='cpu',weights_only=True);m=load_parent();m.load_state_dict(p['state_dict']);del p
 after=test(m,'trained');del m;gc.collect();torch.cuda.empty_cache()
 report={'experiment':experiment,'training':trained,'before':before,'after':after,'parameters':20860416,'limits':experiment['limits']}
 s4.atomic_json(ROOT/'rapport-etape10.json',report)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name!='SAUVEGARDE-VERIFIEE.json']
 s4.atomic_json(ROOT/'SAUVEGARDE-VERIFIEE.json',{'verified_on_mounted_drive':True,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':s4.sha(p)} for p in sorted(files)]})
 print('MANDIA_ETAPE10_TERMINE',json.dumps({'best_step':trained['best_step'],'before':{k:v for k,v in before.items() if k.startswith('fresh_')},'after':{k:v for k,v in after.items() if k.startswith('fresh_')}},ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':main()
