# Mandia N’zassa 21M

Petit modèle Mandia **français → baoulé**, construit et entraîné depuis zéro. **Version de recherche v0.1 : traduction et conversation encore peu fiables.** Le projet s’appelait auparavant MandiaCompact ; le renommage conserve l’historique et les anciens checkpoints.

N’zassa évoque un tissu assemblé à partir de plusieurs morceaux : une image de notre combinaison convolution/GQA et du lien entre les langues. La [Compagnie N’Zassa Danse](https://nzassadanse.wixsite.com/nzassadanse/biographie) documente cette image culturelle akan. Ce choix est symbolique, pas une nouvelle revendication scientifique.

- [Fiche modèle Mandia](https://dataset.mandia.org/gnakalehacker/mandia-nzassa) — catégorie **Modèle**.
- **20 860 416 paramètres**, encodeur2 / décodeur6, FFN1536, vocabulaire4096, plafond512 tokens.
- [Version publique v0.1 : poids et variantes](https://github.com/mandi-model/mandia-nzassa/releases/tag/nzassa-21m-v0.1), code, tokenizer, données préparées, protocoles et rapports dans ce dépôt.
- Export de référence : étape10, sélection validation100. Trois exports étape11 sont fournis pour étudier les échecs ; ils ne remplacent pas la référence.
- AdamW/RNG et les checkpoints complets restent dans le Drive de travail ; aucun dossier Drive n’est publié.

## Résultats récents, avec leurs limites

Trois continuations étape11 de300 mises à jour, graine42, mêmes lots et passages auxiliaires, architecture inchangée. Sélection sur validation : contraste, étape50 ; aucun candidat promu après le test. Les répétitions peuvent diminuer sans amélioration de sens.

| Mesure | Référence étape10 | CE | Contraste | Contraste + UL |
|---|---:|---:|---:|---:|
| chrF++30 nouvelles traductions |11.14|11.32|11.35|13.29|
| chrF++6 nouvelles répliques écrites |10.80|8.31|7.52|7.82|
| Répétition mots4 sur9 vues longues déjà observées |21.9%|17.6%|25.3%|7.2%|
| Correspondances exactes,40 cibles déjà connues |26/40|25/40|25/40|22/40|

39 nouveaux tests issus de4 documents :30 traductions,6 répliques de2 scénarios écrits,3 questions pratiques construites. Français adapté par assistant, baoulé publié copié ; **pas de validation native ni de nouvelles conversations enregistrées**.0/39 correspondances exactes pour chaque modèle ; cela ne mesure pas directement la justesse linguistique. Le diagnostic de source élargi porte sur28 paires de traduction et6 de dialogue ; aucune preuve de compréhension générale. Les9 vues longues partagent2 documents déjà observés. Voir [rapports étape11](reports/stage11) et [fiche technique](MODEL_CARD.md).

## Démarrer

```bash
python -m pip install -r requirements.txt
# Télécharger mandia-nzassa-21m-reference.pt dans la Release v0.1.
python inference.py --weights /chemin/mandia-nzassa-21m-reference.pt --mode translate --text "Les enfants vont à l’école."
```

Le script vérifie les empreintes et charge uniquement les tenseurs avec `weights_only=True`. Il traite les longs textes par segments, sans perdre silencieusement les mots ; ce support technique ne garantit pas la fidélité des traductions. `--mode reply --context '...'` permet un dialogue expérimental ; `--json` affiche les limites/EOS. La génération est gloutonne, sans blocage artificiel des répétitions.

Pour étudier une variante, fournir son manifeste :

```bash
python inference.py --weights /chemin/mandia-nzassa-21m-source-contrastive-ul.pt --manifest weights/source_contrastive_ul-manifest.json --mode translate --text "Votre texte français."
```

Les manifestes étape6/7 restent disponibles pour les exports13M. Les noms de classes et scripts historiques sont conservés pour ne pas casser la reproduction. **Texte uniquement** ; pas de Space d’inférence ou de voix annoncé. Les notebooks sont des outils de calcul facultatifs du projet, pas un dataset.

## État des expériences

La comparaison initiale (étape 4) entraîne deux architectures depuis zéro : hybride 13,78 M et référence GQA 13,80 M. Chacune voit 8 557 013 tokens réels, 6000 mises à jour, graine 42. L'hybride obtient une CE test équilibrée de 3,6496 contre 3,7408. Sur un seul prompt L4 bf16, décodage forcé de 64 tokens et médiane de cinq répétitions : 110,14 contre 89,41 tokens/s ; cache décodeur 271 360 contre 390 144 octets. Ce résultat dépend du protocole ; la référence gagne sur certains domaines. Aucune supériorité générale ni nouveauté architecturale démontrée.

Étape 5 : 393 paires quotidiennes Wiktionary/Kaikki ; le chrF++ traduction sur 52 exemples passe de 9,862 à 14,930, avec 0 correspondance exacte.

Étape 6 : deux continuations de 800 mises à jour comparant CE et CE + contraste de source. La CE gagne sur validation ; meilleurs poids au pas 100. Sur le nouveau test : traduction chrF++ 10,742 → 13,186 (25 vues), réponse 11,728 → 12,704 (15 vues), 0 correspondance exacte. Ces vues viennent de seulement deux documents et réutilisent des références. Les anciens tests régressent légèrement. Les 14 réponses simples apprises restent exactes ; cela ne démontre pas une compréhension générale.

La lecture technique des sorties constate des réponses génériques et des changements de sujet. Aucune évaluation native indépendante. Les documents SMOL sont des scénarios écrits/générés, pas des enregistrements de conversations réelles ; le français de l'étape 6 est adapté par l'assistant et les cibles baoulé sont recopiées des sources publiées.

## Utiliser le modèle

Consulter [MODEL_CARD.md](MODEL_CARD.md) et les empreintes dans [weights/manifest.json](weights/manifest.json). Télécharger le fichier de poids d'inférence depuis les Releases, puis :

```bash
python -m pip install -r requirements.txt
python inference.py --weights /chemin/mandia-compact-stage6-ce.pt --manifest weights/stage6-manifest.json --text 'Bonjour !' --mode reply
```

`reply` produit une réponse ; `translate` vise une traduction. Le modèle est texte uniquement. Il n'effectue pas encore de reconnaissance ni de synthèse vocale.

## Entraîner et reproduire

[notebooks/stage4.ipynb](notebooks/stage4.ipynb) démarre l'expérience initiale avec de nouveaux poids. Les étapes 5 et 6 continuent les checkpoints des étapes précédentes, dans le même Drive. Lire les chemins et cellules avant exécution ; sélectionner un GPU bf16. Les notebooks publics ont leurs sorties effacées : les mesures effectivement obtenues sont dans `reports/`.

Colab est un environnement d'exécution facultatif du projet. **Ce dépôt est un projet de modèle ; les données d'entraînement et leurs crédits sont dans `data/`.** Un Space de démonstration pourra être ajouté ultérieurement ; aucun service public d'inférence Mandia N’zassa n'est annoncé ici.

Les modules historiques sont conservés pour documenter les runs. Leur arborescence de runtime et leurs données sont recréées par les notebooks ; ne pas déplacer les fichiers d'un run en cours ou modifier son code gelé. Les checkpoints complets sont écrits dans Drive tous les 100 pas avec contrôle de signature avant reprise. Le test de reprise CPU vérifie les poids, AdamW, RNG et compteurs ; une égalité bit à bit CUDA n'est pas revendiquée.

## Données, licences et attribution

Le code Mandia est Apache-2.0. **Cette licence ne remplace pas celles des corpus de tiers.** Voir [DATA_SOURCES.md](DATA_SOURCES.md), les fichiers `CREDITS.txt`, la provenance par ligne et les audits. Les données complètes de la Bible et les fichiers audio ne sont pas redistribués ici.

Les données regroupées dans `data/` ne constituent pas un nouveau corpus collecté par Mandia : nous documentons filtrage, séparation, jointures et adaptations. Certaines origines mtBCI ont une licence déclarée sans version précisée. Ne pas interpréter une fiche de code comme une autorisation uniforme de réutilisation de toutes les sources.

## Apprentissage assisté : étape 7 exécutée

Deux continuations de 400 mises à jour : français canonique ou reformulations par l'assistant, mêmes cibles baoulé attestées et mêmes 231 414 tokens supervisés. Les longueurs source diffèrent : 732 256 / 747 192 tokens réels. Le bras enrichi gagne la validation sous garde de régression ; meilleurs poids au pas 50 dans les deux bras.

Sur 40 formulations françaises réservées de **40 cibles déjà connues**, les réponses exactement retrouvées passent de 0/40 avant continuation à 4/40 pour le témoin et 6/40 avec enrichissement. chrF++ : 14,114 → 29,347 / 32,592. Une seule graine et un exercice de réponses avec contexte, sans nouvelles cibles indépendantes : ce résultat ne prouve pas la conversation libre. Le modèle confond encore les personnes, le pain et le sucre ; les deux paires de diagnostic de contexte n'ont aucune paire entièrement exacte. Les anciens tests régressent légèrement.

L'assistant a rédigé les questions et variantes françaises ; il n'a ni inventé les cibles baoulé ni transmis ses poids/logits ou un raisonnement privé. Voir [le bilan](reports/stage7/BILAN-FINAL.txt), [la relecture technique](reports/stage7/RELECTURE-SORTIES.json), [les données et recherches](reports/stage7/RECHERCHE-DONNEES.json) et [la méthode](RESEARCH_NEXT.md).

[Poids QA de recherche étape 7](https://github.com/mandi-model/mandia-nzassa/releases/tag/stage7-qa-research-v1), séparés du modèle étape 6 :

```bash
python inference.py --weights /chemin/mandia-compact-stage7-qa.pt --manifest weights/stage7-qa-manifest.json --context 'Ce que voit Aka est un aigle.' --text 'Peux-tu préciser l’animal qu’Aka aperçoit ?' --mode reply
```

Les checkpoints complets restent dans Drive. Aucun Space de démonstration n'est encore créé.
