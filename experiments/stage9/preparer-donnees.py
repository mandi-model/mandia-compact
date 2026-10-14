"""Source-grounded French augmentation. Never generate Baoule targets.
Run from this workspace; the output preserves historical partitions.
"""
import pathlib,json,hashlib,re,collections,unicodedata,sys
from tokenizers import Tokenizer
BASE=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).resolve().parent
def read(p):return list(map(json.loads,p.read_text().splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(s):return re.sub(r'\W+','',unicodedata.normalize('NFC',s).casefold())
tok=Tokenizer.from_file(str(BASE/'outputs/mandia-compact-public/tokenizer.json'))
parallel=BASE/'outputs/mandia-etape4/parallel-prepare.jsonl'
assert sha(parallel)=='dc2bfd9ba783c5edec159ec35bfbb66206598520bef31f960da3ad92cc240cbd'
old=read(parallel); byid={r['id']:r for r in old}
rows=[];excluded=[]
def add(r):
 if len(tok.encode(r['prompt']).ids)+2>160 or len(tok.encode(r['answer']).ids)+1>192:
  excluded.append({'id':r['id'],'reason':'length; no truncation'});return
 rows.append(r)
def grounded(origin,question,view,domain,answer=None):
 answer=origin['answer'] if answer is None else answer
 assert answer in origin['answer']
 r=dict(origin,id=f"s9-{origin['id']}-{view}",task='reply',domain=domain,
  source_id=origin['id'],context_fr=origin['prompt'],question_fr=question,
  prompt=f"Contexte : {origin['prompt']}\nMessage : {question}",answer=answer,
  original_answer_bci=origin['answer'],answer_span_bci=[origin['answer'].index(answer),origin['answer'].index(answer)+len(answer)],
  teacher_role='French question/instruction only; exact attested Baoule target',
  native_reviewed_by_mandia=False,authentic_recorded_dialogue=False)
 add(r)
# Translation-like contextual instructions are explicit: they are not semantic QA.
for r in old:
 if r['domain']!='gospel':continue
 if len(r['prompt'])<45 or not re.search(r'[.!?»:]',r['prompt']) or re.search(r'^(Voir|Enseignement|Appel|La naissance|Les ancêtres|Jésus commence)',r['prompt']):continue
 grounded(r,'Rapporte ce passage en baoulé, en conservant son sens.','reformulation','bible_reformulation')
# Questions are authored from the French parallel rows. Whole row responses may
# include supporting detail, rather than pretending an unverified BA span is exact.
QUESTIONS={
2236:"Raconte ce que Joseph fait après la naissance de l'enfant.",
2269:"Décris l'habillement et la nourriture de Jean.",
2270:"D'où viennent les personnes qui vont voir Jean ?",
2271:"Que font les gens avant que Jean les baptise et où cela a-t-il lieu ?",
2273:"Quel sort est annoncé pour les arbres qui ne donnent pas de bons fruits ?",
2275:"Que fera-t-il du grain et de la paille dans cette comparaison ?",
2277:"D'où vient Jésus et pourquoi rejoint-il Jean ?",
2278:"Pourquoi Jean s'oppose-t-il au baptême de Jésus dans ce passage ?",
2281:"Que voit Jésus après son baptême ?",
2282:"Qu'annonce la voix venue du ciel ?",
2285:"Pourquoi Jésus est-il conduit dans le désert ?",
2286:"Que ressent Jésus après le jeûne et combien de temps a-t-il duré ?",
2287:"Que demande le tentateur à Jésus au sujet des pierres ?",
2288:"Comment Jésus répond-il à propos du pain et de la parole de Dieu ?",
2290:"Que répond Jésus à propos de mettre Dieu à l'épreuve ?",
2294:"Que fait Jésus lorsqu'il apprend l'emprisonnement de Jean ?",
2295:"Où Jésus s'installe-t-il après Nazareth ?",
2297:"Quel message Jésus commence-t-il à prêcher ?",
2301:"Raconte la rencontre de Jésus avec Simon et André.",
2302:"Quelle invitation Jésus adresse-t-il aux pêcheurs ?",
2303:"Comment les pêcheurs réagissent-ils à l'appel ?",
2304:"Que font Jacques et Jean dans leur barque ?",
2305:"Que laissent les deux frères pour suivre Jésus ?",
2309:"D'où viennent les foules qui suivent Jésus ?",
3027:"Comment ce texte présente-t-il la Bonne Nouvelle ?",
3029:"Quel est le rôle du messager annoncé dans ce passage ?",
3030:"Que crie la voix dans le désert ?",
3031:"Quel appel Jean lance-t-il dans le désert ?",
3032:"D'où viennent les gens baptisés par Jean et que font-ils ?",
3034:"Comment Jean compare-t-il sa puissance à celle de celui qui vient ?",
3035:"Quelle différence Jean fait-il entre les deux baptêmes ?",
3042:"Quel changement de comportement ce message demande-t-il ?",
3049:"Pourquoi l'enseignement de Jésus impressionne-t-il les auditeurs ?",
3052:"Quel ordre Jésus adresse-t-il à l'esprit mauvais ?",
3053:"Comment l'esprit mauvais quitte-t-il l'homme ?",
3055:"Où se répand la renommée de Jésus ?",
3056:"Où vont-ils en quittant la synagogue et avec qui ?",
3058:"Comment Jésus aide-t-il la belle-mère de Simon et que fait-elle ensuite ?",
3059:"Qui amène-t-on auprès de Jésus le soir ?",
3060:"Où la population de la ville se rassemble-t-elle ?",
3069:"Que devient la lèpre de cet homme ?",
3070:"Comment Jésus renvoie-t-il cet homme ?",
3071:"Quelles consignes Jésus donne-t-il à l'homme après sa guérison ?",
2359:"Pourquoi ce passage déconseille-t-il d'accomplir ses devoirs religieux pour être vu ?",
2364:"Comment ce passage recommande-t-il de prier en privé ?",
2365:"Pourquoi ce passage déconseille-t-il de répéter indéfiniment les mêmes paroles ?",
2366:"Que sait déjà le Père avant qu'on lui demande quelque chose ?",
2373:"Pourquoi ne faut-il pas amasser des richesses dans ce monde, selon le passage ?",
2374:"Où le passage conseille-t-il d'amasser ses richesses et pourquoi ?",
2375:"Quel lien le passage établit-il entre le cœur et les richesses ?",
2377:"Quelle comparaison le passage fait-il entre les yeux et une lampe ?",
2380:"Pourquoi le passage dit-il qu'on ne peut pas servir deux maîtres ?",
2381:"Quels sont les deux maîtres qu'on ne peut pas servir ensemble ?",
2384:"Que dit le passage sur les oiseaux et leur nourriture ?",
2387:"Que faut-il rechercher d'abord, selon ce texte ?",
2388:"Quel conseil le passage donne-t-il au sujet du lendemain ?",
2391:"Quel avertissement le passage donne-t-il sur le jugement d'autrui ?",
2395:"Que promet le texte à celui qui demande, cherche et frappe à la porte ?",
2397:"Comment le passage demande-t-il d'agir envers les autres ?",
2399:"Quelle porte faut-il choisir et où conduit le chemin large ?",
2400:"Comment le passage décrit-il le chemin qui mène à la vie ?",
}
for number,q in QUESTIONS.items():
 r=byid[f'mtbci-row-{number}'];assert r['split']=='train' and r['domain']=='gospel'
 grounded(r,q,'question','bible_semantic_qa')
# Include the previously authored pilot questions with their inherited chapter split.
for r in read(BASE/'outputs/mandia-etape8/bible-qa-pilote.jsonl'):
 r=dict(r,id='s9-prior-'+r['id'],domain='bible_semantic_qa');add(r)
# Deduplicate normalized BA responses across chapter partitions, prioritising
# holdouts. This avoids exact-answer leakage without changing chapter membership.
hold={s:{canon(r['answer']) for r in rows if r['split']==s} for s in ['validation','test']}
assert not hold['validation']&hold['test']
rows=[r for r in rows if r['split']!='train' or canon(r['answer']) not in hold['validation']|hold['test']]
bible=rows[:];rows=[]
wiki=read(BASE/'outputs/mandia-etape5/quotidien-prepare.jsonl')
for r in wiki:
 for i,q in enumerate(['Redis cette information en baoulé.','Comment exprimer ce fait en baoulé ?']):
  grounded(r,q,f'context-{i}','daily_contextual_reformulation')
daily=rows[:];rows=[]
smolpath=BASE/'work/recherche-etape6/smoldoc--en_bci.jsonl'
assert sha(smolpath)=='fa56efac8e1df34fec3b81793677d4f33a3f92cfc187e92794dfe2cf9d1e6fdd'
docs=read(smolpath)
# Exact sentence indices/strings retain an audit trail. These are published
# written translated dialogues, NOT recordings or originally French dialogues.
def dialogue(docnum,split,question,indices,context=''):
 d=docs[docnum];tr=[re.sub(r'^(?:A|B|Teacher|Grace)\s*:\s*','',d['trgs'][i]) for i in indices]
 answer=' '.join(tr)
 add({'id':f"s9-smol-{docnum}-{'-'.join(map(str,indices))}",'split':split,'task':'reply','domain':'smol_written_reply',
  'prompt':f'Contexte : {context}\nMessage : {question}' if context else question,
  'question_fr':question,'context_fr':context,'answer':answer,'source_document':d['id'],'source_sentence_indices':indices,
  'source_english_answers':[d['srcs'][i] for i in indices],'source_baoule_answers':[d['trgs'][i] for i in indices],
  'source_url':'https://huggingface.co/datasets/google/smol','source_revision':'fdaff3a1a019f89fa30a562a85d7c1d3e9150444',
  'license':'CC-BY-4.0','french_origin':'assistant translation of source English conversation',
  'baoule_origin':'published translation; exact source sentences joined, explicit role prefixes removed only',
  'native_reviewed_by_mandia':False,'authentic_recorded_dialogue':False,'source_sha256':sha(smolpath)})
dialogue(114,'train',"Savais-tu que la déforestation est l'une des principales causes du changement climatique ?",[1,2])
dialogue(114,'train',"Comment cela fonctionne-t-il ?",[3,4,5],"Nous parlons des effets de la déforestation.")
dialogue(114,'train',"Quels bienfaits les arbres nous apportent-ils ?",[7,8,9,10])
dialogue(114,'train',"La déforestation a donc beaucoup de conséquences négatives.",[12,13])
dialogue(114,'train',"Que pouvons-nous faire pour arrêter la déforestation ?",[17,18,19])
dialogue(12,'train',"Ben doute qu'un point d'interrogation vert tatoué sur le front soit une bonne idée. Que lui répond Tom ?",[4,5])
dialogue(12,'train',"Tom pense que les gens vont trouver son tatouage drôle. Quelle objection Ben lui fait-il ?",[6])
dialogue(12,'train',"Ben dit que les gens vont prendre Tom pour un idiot. Comment Tom le rassure-t-il ?",[7])
dialogue(23,'validation',"Peux-tu me parler un peu de l'histoire du thé ?",[1])
dialogue(23,'validation',"Quand le thé a-t-il été introduit en Europe et par qui ?",[5])
dialogue(23,'validation',"C'est vraiment intéressant, je ne savais pas que le thé avait une si longue histoire.",[9])
dialogue(46,'test',"Maman, qu'est-ce que c'est ?",[1],"L'enfant montre les plantes du salon.")
dialogue(46,'test',"Est-ce que je peux les toucher ?",[3],"L'enfant demande à sa maman s'il peut toucher les plantes.")
dialogue(46,'test',"J'aime bien cette plante.",[7],"L'enfant vient de toucher une feuille douce.")
dialogue(46,'test',"Qu'est-ce que c'est ?",[10,11],"L'enfant montre le câble qui longe le sol.")
dialogue(46,'test',"Est-ce que je peux le toucher ?",[13],"L'enfant demande à sa maman s'il peut toucher le câble électrique.")
dialogues=rows[:]
def save(name,items):
 assert len({r['id'] for r in items})==len(items)
 p=OUT/name;p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items));return sha(p)
hashes={name:save(name,data) for name,data in [('bible-etendue.jsonl',bible),('quotidien-contextuel.jsonl',daily),('dialogues-ecrits-ajout.jsonl',dialogues)]}
# New dialogue docs must never have been in the frozen stage6 corpus.
prior_docs={r['source_document'] for r in read(BASE/'outputs/mandia-etape6/dialogues-prepare.jsonl')}
assert not prior_docs&{r['source_document'] for r in dialogues}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
 assert not {canon(r['answer']) for r in bible if r['split']==a}&{canon(r['answer']) for r in bible if r['split']==b}
 assert not {r['source_document'] for r in dialogues if r['split']==a}&{r['source_document'] for r in dialogues if r['split']==b}
audit={'sha256':hashes,'counts':{k:dict(collections.Counter(r['split'] for r in v)) for k,v in [('bible',bible),('daily',daily),('written_dialogues',dialogues)]},
 'bible_source_rows':len({r.get('source_id') for r in bible}),'bible_domains':dict(collections.Counter(r['domain'] for r in bible)),
 'historical_chapter_and_wiki_partitions_preserved':True,'new_dialogue_documents_never_used_in_stage6':True,
 'bible_normalized_targets_disjoint_between_splits':True,'length_exclusions':excluded,
 'authentic_recorded_new_dialogues':0,'native_validation':False,
 'limits':['Bible translations are existing mtBCI rows only, edition/verse identifiers unspecified; no new full-Bible import',
 'most expanded Bible views are contextual reformulation, not independently verified semantic reasoning',
 'daily rephrasing views are not new unique facts', 'new SMOL replies are translated written dialogues; French authored from English',
 'known40 daily QA is already observed and uses known target families; report as regression only']}
(OUT/'AUDIT-DONNEES.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='length_exclusions'},ensure_ascii=False,indent=2));print('excluded',len(excluded))
