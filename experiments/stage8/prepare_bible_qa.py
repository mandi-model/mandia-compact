"""Small grounded QA pilot: existing mtBCI rows, fixed chapter split, no new translation."""
import json, pathlib, hashlib, collections, unicodedata, re
BASE=pathlib.Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/mandia-etape8'
SOURCE=BASE/'outputs/mandia-etape4/parallel-prepare.jsonl'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(SOURCE)=='dc2bfd9ba783c5edec159ec35bfbb66206598520bef31f960da3ad92cc240cbd'
source={r['id']:r for r in map(json.loads,SOURCE.read_text().splitlines())}
# Questions authored in French. Optional answer is an EXACT contiguous Baoule span.
# These are textual Bible QA, never labelled as recorded daily conversations.
train=[
 (3028,'Dans quel livre ce passage dit-il que ces paroles sont écrites ?','Quel livre de prophète est mentionné ici ?',None),
 (3033,'Que mangeait Jean, selon ce passage ?','Quelle était la nourriture de Jean dans ce texte ?',"Lalo nin awɛma nzue yɛ ɔ di-ɔ ."),
 (3033,'Comment Jean était-il habillé ?','Décris les vêtements de Jean mentionnés dans le passage.',"Nyɔnngɔnmɛn ndrɛ tralɛ o Zan wun, yɛ kplo kun ci i afiɛn."),
 (3036,'Raconte la venue et le baptême de Jésus dans ce passage.','Que dit ce texte sur le lieu de départ et le baptême de Jésus ?',None),
 (3037,"Qu'a vu Jésus en sortant de l'eau ?","Décris ce que Jésus voit au moment où il sort de l'eau.",None),
 (3038,"Que dit la voix venant du ciel ?","Rapporte les paroles entendues du ciel dans ce texte.",None),
 (3039,"Où l'Esprit conduit-il Jésus ?","Que fait l'Esprit tout de suite après, selon ce passage ?",None),
 (3040,'Décris le séjour de Jésus dans le désert.','Que se passe-t-il pendant les quarante jours de Jésus dans le désert ?',None),
 (3041,"Que fait Jésus après la mise en prison de Jean ?","Après l'emprisonnement de Jean, où Jésus va-t-il et que proclame-t-il ?",None),
 (3043,'Que font Simon et André lorsque Jésus les voit ?','Comment le passage décrit-il la rencontre de Jésus avec Simon et André ?',None),
 (3044,'Que propose Jésus aux pêcheurs ?','Quelles paroles Jésus adresse-t-il aux pêcheurs ici ?',None),
 (3045,'Comment les pêcheurs réagissent-ils immédiatement ?','Que laissent-ils pour suivre Jésus ?',None),
 (3046,'Que font Jacques et Jean dans leur barque ?','Que voit Jésus un peu plus loin, dans ce passage ?',None),
 (3047,"Que font les fils de Zébédée après l'appel de Jésus ?","Qui laissent-ils dans la barque lorsqu'ils suivent Jésus ?",None),
 (3048,'Que font Jésus et ses disciples à Capernaüm ?','Décris ce qui se passe à Capernaüm au jour du sabbat.',None),
 (3057,'Quel problème de santé a la belle-mère de Simon dans ce récit ?','Dans quel état se trouve la belle-mère de Simon quand ils arrivent ?',None),
 (3062,'Que fait Jésus très tôt le lendemain ?','Raconte où Jésus va prier au petit matin.',None),
 (3063,'Qui part à la recherche de Jésus ?','Quels personnages cherchent Jésus dans ce passage ?',None),
 (3064,'Que disent-ils à Jésus après l’avoir trouvé ?','Quelles paroles lui adressent-ils quand ils le trouvent ?',None),
 (3065,'Pourquoi Jésus veut-il aller dans les villages voisins ?','Que propose Jésus à ses compagnons et pour quelle raison ?',None),
 (3066,'Que fait Jésus dans toute la Galilée ?','Comment le texte décrit-il les activités de Jésus en Galilée ?',None),
 (3067,'Que demande le lépreux à Jésus ?','Comment le lépreux sollicite-t-il Jésus dans ce récit ?',None),
 (3068,'Comment Jésus répond-il à la demande du lépreux ?','Quels gestes et quelles paroles Jésus adresse-t-il au lépreux ?',None),
 (2279,'Comment Jésus répond-il à Jean dans ce passage ?','Quelle raison Jésus donne-t-il pour que Jean accepte ?',None),
]
validation=[
 (3083,"Que font les fidèles à l'extérieur pendant que l'encens brûle ?",None,None),
 (3085,"Comment Zacharie réagit-il en voyant l'ange ?",None,None),
 (3086,"Que dit l'ange à Zacharie à propos de sa prière et de son enfant ?",None,None),
 (3087,'Que provoquera la naissance de cet enfant ?',None,None),
 (3091,"Pourquoi Zacharie demande-t-il comment il pourra croire cette annonce ?",None,None),
 (3092,"Comment l'ange se présente-t-il et explique-t-il sa mission ?",None,None),
 (3096,'Que fait Zacharie quand son service au temple est terminé ?',None,None),
 (3099,"Où Dieu envoie-t-il Gabriel au sixième mois ?",None,None),
]
test=[
 (2239,'Raconte où Jésus naît et où arrivent les savants dans ce passage.',None,None),
 (2248,"Que ressentent les savants en voyant l'étoile arrêtée ?",None,None),
 (2249,'Que voient les savants en entrant dans la maison ?',None,None),
 (2250,"Comment les savants rendent-ils hommage à l'enfant ?",None,"Be kotoli i bo, be mɛnmɛnnin i."),
 (2250,"Quels cadeaux offrent les savants après avoir ouvert leurs bagages ?",None,"Be nyannjili be trɔ mun, be cɛli i sika ɔkwlɛ, nin ansans, ɔ nin mir ."),
 (2251,'Pourquoi les savants prennent-ils un autre chemin pour rentrer ?',None,None),
 (2254,"Que fait Joseph pendant la nuit après l'avertissement ?",None,None),
 (2255,"Jusqu'à quel événement Joseph reste-t-il en Égypte ?",None,None),
]
rows=[]
for split,items in [('train',train),('validation',validation),('test',test)]:
 for number,q1,q2,span in items:
  origin=source[f'mtbci-row-{number}']; assert origin['split']==split and origin['domain']=='gospel'
  answer=origin['answer'] if span is None else span
  assert answer and answer in origin['answer']
  start=origin['answer'].index(answer)
  for view,q in enumerate([q1,q2] if q2 else [q1]):
   rows.append({'id':f'bibleqa-{number}-{items.index((number,q1,q2,span))}-{view}',
    'family_id':hashlib.sha256((origin['id']+answer).encode()).hexdigest()[:16],
    'split':split,'task':'reply','domain':'bible_qa','source_dataset':origin['source_dataset'],
    'source_url':'https://github.com/Kjeanclaude/mtBCI-1.0-Corpus',
    'source_id':origin['id'],'source_document':origin['source_document'],'source_row_1based':number,
    'source_parallel_sha256':sha(SOURCE),'source_group_id':origin['group_id'],
    'context_fr':origin['prompt'],'question_fr':q,
    'prompt':f"Contexte : {origin['prompt']}\nMessage : {q}",
    'answer':answer,'answer_span_bci':[start,start+len(answer)],'original_answer_bci':origin['answer'],
    'teacher_role':'French question only, based on supplied parallel row; no generated Baoule',
    'answer_type':'whole_aligned_row' if span is None else 'contiguous_source_span',
    'native_reviewed_by_mandia':False,'authentic_recorded_dialogue':False,
    'verse_id':None,'edition_id':None,'provenance_limit':'mtBCI author research corpus; Bible edition and verse identifiers unspecified; no new permission asserted for full Bible',
    'split_scope':'inherited chapter split, not necessarily unseen story; Gospel translation holdouts already observed in previous regression checks'})
def canon(s):return re.sub(r'\W+','',unicodedata.normalize('NFC',s).casefold())
sets={s:{canon(r['answer']) for r in rows if r['split']==s} for s in ['train','validation','test']}
assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
assert len({r['id'] for r in rows})==len(rows)
OUT.mkdir(parents=True,exist_ok=True)
p=OUT/'bible-qa-pilote.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
audit={'rows':len(rows),'counts':dict(collections.Counter(r['split'] for r in rows)),
 'unique_target_families':len({r['family_id'] for r in rows}),
 'unique_original_rows':len({r['source_id'] for r in rows}),
 'chapters':{s:sorted({r['source_document'] for r in rows if r['split']==s}) for s in sets},
 'all_answers_exact_source_spans':True,'original_chapter_splits_preserved':True,
 'normalized_answer_overlap_between_splits':False,'native_validation':False,
 'newly_collected_authentic_daily_dialogues':0,'bible_qa_is_not_daily_dialogue':True,
 'qa_test_is_new_but_inherited_translation_holdouts_have_been_evaluated_before':True,
 'data_sha256':sha(p),'source_sha256':sha(SOURCE)}
(OUT/'AUDIT-BIBLE-QA.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False,indent=2))
