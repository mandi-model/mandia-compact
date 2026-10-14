# MandiaCompact S2S

Projet de recherche Mandia : comprendre une entrée française et produire du baoulé avec un petit modèle construit et entraîné depuis zéro. **Prototype expérimental, conversation générale encore peu fiable.**

- **Fiche modèle** : https://dataset.mandia.org/gnakalehacker/mandia-compact
- **13 782 528 paramètres**, encodeur-décodeur hybride convolution causale / GQA, tokenizer ByteBPE de 4096 tokens.
- Code et rapports dans ce dépôt ; poids d'inférence distribués séparément dans les Releases lorsqu'ils sont disponibles.
- Les checkpoints complets de travail (poids, AdamW, états RNG et compteurs) restent sauvegardés dans Google Drive. Aucun dossier Drive personnel n'est rendu public par ce dépôt.

## État des expériences

La comparaison initiale (étape 4) entraîne deux architectures depuis zéro : hybride 13,78 M et référence GQA 13,80 M. Chacune voit 8 557 013 tokens réels, 6000 mises à jour, graine 42. L'hybride obtient une CE test équilibrée de 3,6496 contre 3,7408. Sur un seul prompt L4 bf16, décodage forcé de 64 tokens et médiane de cinq répétitions : 110,14 contre 89,41 tokens/s ; cache décodeur 271 360 contre 390 144 octets. Ce résultat dépend du protocole ; la référence gagne sur certains domaines. Aucune supériorité générale ni nouveauté architecturale démontrée.

Étape 5 : 393 paires quotidiennes Wiktionary/Kaikki ; le chrF++ traduction sur 52 exemples passe de 9,862 à 14,930, avec 0 correspondance exacte.

Étape 6 : deux continuations de 800 mises à jour comparant CE et CE + contraste de source. La CE gagne sur validation ; meilleurs poids au pas 100. Sur le nouveau test : traduction chrF++ 10,742 → 13,186 (25 vues), réponse 11,728 → 12,704 (15 vues), 0 correspondance exacte. Ces vues viennent de seulement deux documents et réutilisent des références. Les anciens tests régressent légèrement. Les 14 réponses simples apprises restent exactes ; cela ne démontre pas une compréhension générale.

La lecture technique des sorties constate des réponses génériques et des changements de sujet. Aucune évaluation native indépendante. Les documents SMOL sont des scénarios écrits/générés, pas des enregistrements de conversations réelles ; le français de l'étape 6 est adapté par l'assistant et les cibles baoulé sont recopiées des sources publiées.

## Utiliser le modèle

Consulter [MODEL_CARD.md](MODEL_CARD.md) et les empreintes dans [weights/manifest.json](weights/manifest.json). Télécharger le fichier de poids d'inférence depuis les Releases, puis :

```bash
python -m pip install -r requirements.txt
python inference.py --weights /chemin/mandia-compact-stage6-ce.pt --text 'Bonjour !' --mode reply
```

`reply` produit une réponse ; `translate` vise une traduction. Le modèle est texte uniquement. Il n'effectue pas encore de reconnaissance ni de synthèse vocale.

## Entraîner et reproduire

[notebooks/stage4.ipynb](notebooks/stage4.ipynb) démarre l'expérience initiale avec de nouveaux poids. Les étapes 5 et 6 continuent les checkpoints des étapes précédentes, dans le même Drive. Lire les chemins et cellules avant exécution ; sélectionner un GPU bf16. Les notebooks publics ont leurs sorties effacées : les mesures effectivement obtenues sont dans `reports/`.

Colab est un environnement d'exécution facultatif du projet. **Ce dépôt est un projet de modèle ; les données d'entraînement et leurs crédits sont dans `data/`.** Un Space de démonstration pourra être ajouté ultérieurement ; aucun service public d'inférence MandiaCompact n'est annoncé ici.

Les modules historiques sont conservés pour documenter les runs. Leur arborescence de runtime et leurs données sont recréées par les notebooks ; ne pas déplacer les fichiers d'un run en cours ou modifier son code gelé. Les checkpoints complets sont écrits dans Drive tous les 100 pas avec contrôle de signature avant reprise. Le test de reprise CPU vérifie les poids, AdamW, RNG et compteurs ; une égalité bit à bit CUDA n'est pas revendiquée.

## Données, licences et attribution

Le code Mandia est Apache-2.0. **Cette licence ne remplace pas celles des corpus de tiers.** Voir [DATA_SOURCES.md](DATA_SOURCES.md), les fichiers `CREDITS.txt`, la provenance par ligne et les audits. Les données complètes de la Bible et les fichiers audio ne sont pas redistribués ici.

Les données regroupées dans `data/` ne constituent pas un nouveau corpus collecté par Mandia : nous documentons filtrage, séparation, jointures et adaptations. Certaines origines mtBCI ont une licence déclarée sans version précisée. Ne pas interpréter une fiche de code comme une autorisation uniforme de réutilisation de toutes les sources.

## Apprentissage assisté par un grand modèle

La prochaine piste est une expérience contrôlée de reformulations françaises et de cibles baoulé attestées. Ce n'est pas encore un résultat entraîné. [RESEARCH_NEXT.md](RESEARCH_NEXT.md) précise protocole, sources et limites : aucun accès aux poids ou logits privés de l'assistant ; aucune traduction baoulé inventée présentée comme vérifiée.
