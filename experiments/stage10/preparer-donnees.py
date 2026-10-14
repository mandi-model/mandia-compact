"""French authored from public English; every Baoule target is source text.
New source families held out, no known40-based selection. Workspace preparation.
"""
import pathlib,json,hashlib,collections,re,unicodedata
from tokenizers import Tokenizer
BASE=pathlib.Path(__file__).resolve().parents[2];OUT=pathlib.Path(__file__).resolve().parent
def read(p):return list(map(json.loads,p.read_text().splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(s):return re.sub(r'\W+','',unicodedata.normalize('NFC',s).casefold())
rawsent=BASE/'work/recherche-etape6/smolsent--en_bci.jsonl';rawdoc=BASE/'work/recherche-etape6/smoldoc--en_bci.jsonl'
assert sha(rawsent)=='f5250fee53b2bd297a4fa7f818514987cbb320e03fbccd28dc7696f9360adfe6'
assert sha(rawdoc)=='fa56efac8e1df34fec3b81793677d4f33a3f92cfc187e92794dfe2cf9d1e6fdd'
sent={r['id']:r for r in read(rawsent)};docs=read(rawdoc)
tok=Tokenizer.from_file(str(BASE/'outputs/mandia-compact-public/tokenizer.json'))
prior=[]
for f in ['mandia-etape4/parallel-prepare.jsonl','mandia-etape4/replies-prepare.jsonl','mandia-etape5/quotidien-prepare.jsonl','mandia-etape6/dialogues-prepare.jsonl','mandia-etape7/teacher-grounded-qa.jsonl','mandia-etape8/bible-qa-pilote.jsonl','mandia-etape9/bible-etendue.jsonl','mandia-etape9/quotidien-contextuel.jsonl','mandia-etape9/dialogues-ecrits-ajout.jsonl']:
 prior+=read(BASE/'outputs'/f)
prior_targets={canon(r['answer']) for r in prior};prior_docs={r.get('source_document') for r in prior if r.get('source_document')}
rows=[];excluded=[]
def add(r):
 r=dict(r,source_url='https://huggingface.co/datasets/google/smol',source_revision='fdaff3a1a019f89fa30a562a85d7c1d3e9150444',license='CC-BY-4.0',
  french_origin='assistant translation of published English, semantically screened; not a human French corpus',
  baoule_origin='published translation preserved; no assistant-generated Baoule',native_reviewed_by_mandia=False,authentic_recorded_dialogue=False)
 r['source_tokens']=len(tok.encode(r['prompt']).ids)+2;r['target_tokens']=len(tok.encode(r['answer']).ids)+1
 if r['source_tokens']>512 or r['target_tokens']>512:
  excluded.append({'id':r['id'],'reason':'length; no truncation','source_tokens':r['source_tokens'],'target_tokens':r['target_tokens']});return
 assert r['prompt'].strip() and r['answer'].strip()
 rows.append(r)
french={}
for line in (OUT/'francais-sentences.tsv').read_text().splitlines():
 i,fr=line.split('\t',1);assert int(i) not in french;french[int(i)]=fr
eligible=[]
for i,fr in french.items():
 r=sent[i]
 if canon(r['trg']) in prior_targets:
  excluded.append({'id':f's10-sent-{i}','reason':'target already present in prior corpus; do not reassign holdouts'});continue
 eligible.append(i)
ordered=sorted(eligible,key=lambda i:hashlib.sha256(('mandia10-new-sent-'+str(i)).encode()).hexdigest())
splits={i:'validation' if j<12 else 'test' if j<24 else 'train' for j,i in enumerate(ordered)}
for i in eligible:
 r=sent[i];add({'id':f's10-sent-{i}','source_family':f'smolsent-{i}','source_document':f'smolsent-{i}',
  'split':splits[i],'domain':'new_sentence_translation','task':'translate','prompt':french[i],'answer':r['trg'],
  'source_english':r['src'],'source_baoule':r['trg'],'source_raw_sha256':sha(rawsent)})
frdocs={int(i):r for i,r in json.loads((OUT/'francais-documents.json').read_text()).items()}
partitions={0:'train',49:'train',104:'train',70:'train',47:'validation',73:'test',97:'test',7:'train',28:'test',113:'test'}
for i,split in partitions.items():assert docs[i]['id'] not in prior_docs
for i in [0,49,104,70,47,73,97]:
 d=docs[i];fr=frdocs[i];assert len(fr)==len(d['srcs'])==len(d['trgs'])
 for j,(p,a) in enumerate(zip(fr,d['trgs'])):
  add({'id':f's10-doc-{i}-sentence-{j}','source_family':d['id'],'source_document':d['id'],'split':partitions[i],'task':'translate','domain':'new_document_sentence',
   'prompt':p,'answer':a,'source_english':d['srcs'][j],'source_baoule':a,'source_indices':[j],'source_raw_sha256':sha(rawdoc)})
 # Aligned consecutive multi-sentence blocks; original document partition retained.
 for start in range(0,len(fr),4):
  ix=list(range(start,min(start+4,len(fr))))
  if len(ix)<2:continue
  add({'id':f's10-doc-{i}-paragraph-{start}','source_family':d['id'],'source_document':d['id'],'split':partitions[i],'task':'translate','domain':'new_paragraph_translation',
   'prompt':' '.join(fr[j] for j in ix),'answer':' '.join(d['trgs'][j] for j in ix),
   'source_english':[d['srcs'][j] for j in ix],'source_baoule':[d['trgs'][j] for j in ix],'source_indices':ix,'source_raw_sha256':sha(rawdoc),
    'construction':'aligned consecutive source sentences joined with spaces; not new independent content'})
 add({'id':f's10-doc-{i}-whole','source_family':d['id'],'source_document':d['id'],'split':partitions[i],'task':'translate','domain':'whole_document_translation',
  'prompt':' '.join(fr),'answer':' '.join(d['trgs']),'source_english':d['srcs'],'source_baoule':d['trgs'],'source_indices':list(range(len(fr))),'source_raw_sha256':sha(rawdoc),
  'construction':'whole aligned document joined with spaces; content overlaps sentence and paragraph views within same partition only'})
def frstrip(s):return re.sub(r'^(?:A|B|Patient|Médecin|Dentiste)\s*:\s*','',s)
def bastrip(s):
 return s.split(':',1)[1].strip() if re.match(r'^(?:A|B|Tupkatʃɛ fuɛ|Dɔrtrɔ bian|Be je dɔrtrɔ bian)\s*:',s) else s
def reply(i,prompt_ix,answer_ix,history_ix):
 d=docs[i];fr=frdocs[i]
 context='Dialogue écrit. '+' '.join(fr[j] for j in history_ix) if history_ix else 'Dialogue écrit.'
 q=' '.join(frstrip(fr[j]) for j in prompt_ix);a=' '.join(bastrip(d['trgs'][j]) for j in answer_ix)
 add({'id':f's10-reply-{i}-{answer_ix[0]}','source_family':d['id'],'source_document':d['id'],'split':partitions[i],
  'task':'reply','domain':'new_written_next_turn','prompt':f'Contexte : {context}\nMessage : {q}','question_fr':q,'context_fr':context,'answer':a,
  'source_prompt_indices':prompt_ix,'source_answer_indices':answer_ix,'source_history_indices':history_ix,
  'source_english_prompt':[d['srcs'][j] for j in prompt_ix],'source_english_answer':[d['srcs'][j] for j in answer_ix],
  'source_baoule':[d['trgs'][j] for j in answer_ix],'source_raw_sha256':sha(rawdoc),
  'construction':'next-turn source reply, explicit role prefix removed only; not newly recorded dialogue'})
# Only symptom-description/follow-up turns, not the diagnosis or medication claims.
for p,a,h in [([0,1,2],[3],[]),([4],[5],[0,3]),([6],[7],[4,5]),([8],[9],[6,7]),([10],[11],[8,9])]:reply(7,p,a,h)
reply(28,[2],[3],[0,1]);reply(28,[6],[7],[3,5])
# Omit visibly doubtful alignment and cultural claim lines 3 and 7.
reply(113,[0],[1,2],[]);reply(113,[4],[5,6],[0,1,2]);reply(113,[9],[10,11],[4,5,6])
# Validation next-turn examples from the held-out birthday document are explicitly
# authored questions, not claimed source conversational turns.
for j,q in [(3,'Comment commencer à préparer mon anniversaire ?'),(6,'Que puis-je faire ensuite pour préparer la prochaine année ?'),(9,'Et comment préparer le jour de mon anniversaire ?')]:
 d=docs[47];add({'id':f's10-birthday-qa-{j}','source_family':d['id'],'source_document':d['id'],'split':'validation','task':'reply','domain':'grounded_advice_qa',
  'prompt':q,'answer':d['trgs'][j],'source_english':d['srcs'][j],'source_baoule':d['trgs'][j],'source_indices':[j],
  'construction':'assistant French question with published source response; not authentic next-turn dialogue','source_raw_sha256':sha(rawdoc)})
assert len({r['id'] for r in rows})==len(rows)
# No source family or normalized target shared across partitions. Previously
# observed doc46 and known40 are never admitted into this new-data corpus.
for a,b in [('train','validation'),('train','test'),('validation','test')]:
 assert not {r['source_family'] for r in rows if r['split']==a}&{r['source_family'] for r in rows if r['split']==b}
 assert not {canon(r['answer']) for r in rows if r['split']==a}&{canon(r['answer']) for r in rows if r['split']==b}
for r in rows:
 if r['split']!='train':assert canon(r['answer']) not in {canon(x['answer']) for x in prior if x['split']=='train'},r['id']
path=OUT/'nouveaux-paralleles.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
audit={'counts':{s:dict(collections.Counter(r['domain'] for r in rows if r['split']==s)) for s in ['train','validation','test']},
 'rows':len(rows),'new_sentence_families':len(eligible),'source_doc_partitions':{str(i):partitions[i] for i in partitions},
 'source_family_overlap_between_splits':False,'normalized_target_overlap_between_splits':False,'new_holdout_target_seen_in_prior_train':False,
 'max_source_tokens':max(r['source_tokens'] for r in rows),'max_target_tokens':max(r['target_tokens'] for r in rows),
 'paragraph_lengths':{s:[{'id':r['id'],'french_words':len(r['prompt'].split()),'source_tokens':r['source_tokens'],'baoule_words':len(r['answer'].split()),'target_tokens':r['target_tokens']} for r in rows if r['split']==s and r['domain']=='new_paragraph_translation'] for s in ['train','validation','test']},
 'excluded':excluded,'data_sha256':sha(path),'raw_sha256':{'smolsent':sha(rawsent),'smoldoc':sha(rawdoc)},
 'native_validation':False,'authentic_recorded_new_dialogues':0,
 'limits':['French translated by assistant from English, not native reviewed','SMOL is published translated text, with spelling variation and potential alignment errors','5 fresh dialogue test replies from only 2 written source documents, not spontaneous recorded conversations','paragraphs share content with sentences within each same partition; no independent-content inflation','birthday reply validation is authored advice QA, not an authentic conversational validation set','no claim of reliable translation of arbitrary long French text']}
(OUT/'AUDIT-DONNEES.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['paragraph_lengths','excluded']},ensure_ascii=False,indent=2));print('excluded',len(excluded));print('paragraph target lengths TRAIN', [r['target_tokens'] for r in rows if r['split']=='train' and r['domain']=='new_paragraph_translation'])
