import json,pathlib,hashlib,re,unicodedata,collections
from tokenizers import Tokenizer
O=pathlib.Path(__file__).resolve().parent;B=O.parents[1]
raw=B/'outputs/mandia-etape10/source-smoldoc-en-bci.jsonl';docs=[json.loads(x) for x in raw.read_text().splitlines()]
tok=Tokenizer.from_file(str(B/'outputs/mandia-compact-public/tokenizer.json'))
fr18=["Beaucoup de gens se lèvent avant le lever du soleil pour commencer leur journée de travail.","Les agriculteurs peuvent se lever avant l'aube pour traire leurs vaches ou s'occuper de leurs cultures.","Les pêcheurs peuvent partir en mer avant le lever du soleil.","Les personnes qui travaillent en ville peuvent aussi se lever tôt pour prendre le bus ou le train et aller au travail.","Une fois levés, les gens prennent généralement un petit déjeuner simple : de la bouillie, du pain ou du thé.","Ensuite, ils se préparent pour la journée.","En Gambie, beaucoup de gens portent des vêtements traditionnels, par exemple de longues robes ou jupes pour les femmes, et des pantalons longs ou des chemises pour les hommes.","Ils peuvent aussi porter un foulard sur la tête ou une casquette.","Après le petit déjeuner, chacun vaque à ses activités quotidiennes.","Les enfants vont à l'école, tandis que les adultes vont au travail.","En Gambie, beaucoup de gens travaillent dans l'agriculture, la pêche ou le tourisme.","D'autres travaillent dans l'administration, le commerce ou l'enseignement.","L'après-midi, les gens peuvent faire une pause dans leur travail pour se détendre ou discuter avec d'autres personnes.","Ils peuvent se promener, rendre visite à leurs amis et à leur famille, ou jouer.","Le soir, les gens dînent souvent en famille.","Le dîner est généralement un repas simple composé de riz, de légumes et de viande ou de poisson."]
fr34=["J'étais sur Internet, tranquillement occupé à mes affaires, quand j'ai vu une publicité dans une fenêtre surgissante qui disait que mon ordinateur avait un virus.","Je me suis dit : Quoi ? Impossible !","Mais j'ai cliqué dessus ; elle m'a envoyé vers un site qui paraissait tout à fait officiel et qui disait que je devais télécharger ce logiciel pour supprimer le virus.","Alors je l'ai fait et je l'ai installé ; il a ensuite commencé à remplir mon ordinateur de logiciels malveillants.","J'étais furieux !","J'ai essayé de le désinstaller, mais il ne me laissait pas faire.","J'ai finalement dû apporter mon ordinateur chez un réparateur ; ils ont dû tout effacer et réinstaller Windows.","C'était un vrai cauchemar.","Voilà pourquoi je fais toujours très attention avant de cliquer sur les publicités dans des fenêtres surgissantes.","On ne sait jamais ce qu'elles vont faire.","Elles pourraient contenir des logiciels malveillants ou voler vos renseignements personnels.","Cela ne vaut vraiment pas la peine de prendre ce risque.","Si vous voyez une publicité suspecte dans une fenêtre surgissante, fermez-la simplement et ne cliquez pas dessus.","Et si vous n'êtes pas certain que ce soit sans danger, abstenez-vous simplement."]
rows=[]
def add(i,ix,prompt,task,domain,englishprompt=None):
 d=docs[i];a=' '.join(d['trgs'][j] for j in ix)
 r=dict(id=f's11-{domain}-{i}-{ix[0]}',split='test',task=task,domain=domain,prompt=prompt,answer=a,source_document=d['id'],source_family=d['id'],source_indices=ix,source_english=[d['srcs'][j] for j in ix],source_baoule=[d['trgs'][j] for j in ix],source_english_prompt=englishprompt,source_url='https://huggingface.co/datasets/google/smol',source_revision='fdaff3a1a019f89fa30a562a85d7c1d3e9150444',source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),license='CC-BY-4.0',french_origin='assistant translated/adapted source English',baoule_origin='exact published target(s) joined; no generated positive Baoule',native_reviewed_by_mandia=False,authentic_recorded_dialogue=False)
 r['source_tokens']=len(tok.encode(prompt).ids)+2;r['target_tokens']=len(tok.encode(a).ids)+1
 assert max(r['source_tokens'],r['target_tokens'])<=512
 rows.append(r)
for i,fr in [(18,fr18),(34,fr34)]:
 assert len(fr)==len(docs[i]['trgs'])
 for j,p in enumerate(fr):add(i,[j],p,'translate','fresh_translation')
# Only these direct next-turn portions: no insults as desired response and no Hemingway identity.
for q,j,en in [("Tu m'as coupé la route ! Tu as failli me faire avoir un accident !",3,[2]),("Tu aurais au moins pu mettre ton clignotant !",5,[4]),("Non, tu ne l'as pas mis ! Tu t'es déporté juste devant moi !",7,[6])]:
 add(22,[j],'Contexte : Discussion entre deux automobilistes après une manœuvre dangereuse.\nMessage : '+q,'reply','fresh_written_reply',[docs[22]['srcs'][k] for k in en])
for q,j,en in [("Oui, mais je n'ai pas commandé un filet mignon cru !",5,[4]),("Monsieur, tous nos steaks sont servis crus. Nous demandons à nos clients de les cuire eux-mêmes.",6,[5]),("Mais je ne sais pas cuisiner !",7,[6])]:
 add(62,[j],'Contexte : Dialogue comique fictif entre un client et un serveur dans un club de comédie qui sert des steaks crus.\nMessage : '+q,'reply','fresh_written_reply',[docs[62]['srcs'][k] for k in en])
for q,j in [("Une publicité suspecte vient de s'ouvrir sur mon ordinateur. Que dois-je faire ?",12),("Je ne sais pas si je peux cliquer sur cette publicité sans risque. Que me conseilles-tu ?",13),("Quels risques peuvent présenter ces publicités ?",10)]:
 add(34,[j],q,'reply','fresh_authored_advice')
canon=lambda s:re.sub(r'\W+','',unicodedata.normalize('NFC',s).casefold())
prior=[]
for d in range(4,11):
 for p in (B/f'outputs/mandia-etape{d}').glob('*.jsonl'):
  try:prior += [r for r in map(json.loads,p.read_text().splitlines()) if 'answer' in r and 'split' in r]
  except Exception:pass
prior_docs={r.get('source_document') for r in prior};assert not prior_docs & {r['source_document'] for r in rows}
trainanswers={canon(r['answer']) for r in prior if r['split']=='train'}
assert not trainanswers & {canon(r['answer']) for r in rows}
p=O/'nouveaux-tests.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
audit=dict(rows=len(rows),counts=dict(collections.Counter(r['domain'] for r in rows)),source_documents=sorted({r['source_document'] for r in rows}),source_documents_seen_in_prior_prepared_sets=False,targets_seen_in_prior_train=False,test_not_used_for_training_or_selection=True,data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),limits=['written/generated SMOL source, not spontaneous recorded conversations','French assistant-authored; Baoule copied exactly, no native review','six direct replies from only two written source documents, road dispute and fictional comedy','three computer advice prompts constructed from exact translated monologue responses; not authentic dialogue','translation and advice from doc34 overlap within test only','all previously observed tests are regression tests, not fresh tests'])
(O/'AUDIT-NOUVEAUX-TESTS.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit,ensure_ascii=False))
