# MandiaCompact S2S — fiche modèle

## Objectif et statut
Français → baoulé (`bci`), traduction et réponse courte. Poids propres Mandia, initialisation aléatoire à l'étape 4, continuations 5 et 6. Candidat actuel : étape 6, CE, sélection sur validation au pas 100. Il ne répond pas encore correctement de façon générale.

## Architecture
Encodeur bidirectionnel 2 blocs GQA ; décodeur 6 blocs avec cross-attention au français dans chaque bloc, dont 4 blocs de convolution causale et 2 de GQA. Dimension 384, 6 têtes de requête, 2 têtes K/V, FFN 768, RMSNorm, RoPE, SwiGLU, embeddings et projection de sortie partagés. 13 782 528 paramètres. ByteBPE 4096 appris sur train seulement. Les composants sont issus de méthodes connues.

## Entraînement et sélection
Étape 4 : 6000 mises à jour depuis zéro, même budget de tokens que la référence GQA, meilleurs poids au pas 1250. Étape 5 : continuation 600 pas, meilleur pas 400. Étape 6 : deux bras 800 pas, meilleur CE au pas 100, AdamW neuf pour chaque continuation. Les checkpoints finaux de reprise et les meilleurs poids de validation sont distincts. Les détails et limites sont dans les rapports originaux.

## Données et évaluation
mtBCI, Google SMOL/GATITOS, transcriptions WAXAL et exemples Wiktionary/Kaikki. Voir DATA_SOURCES.md. Aucune Bible complète ni audio entraîné. Évaluation automatique CE, chrF++, exactitude des références ; petite évaluation, une graine, aucune validation native. Pas de comparaison des chrF++ entre corpus différents. Les tests antérieurs déjà observés sont des contrôles de régression, pas des tests nouveaux.

Sur le nouveau test étape 6 : chrF++ traduction 13,186 sur 25 vues, réponse 12,704 sur 15 vues, exactitude 0/25 et 0/15. 14/14 références simples apprises conservées. Les 40 vues ne sont pas 40 références indépendantes. Les sorties restent souvent hors sujet.

## Usage et limites
Recherche, prototypage et contributions linguistiques. Pas de garantie de justesse, de naturalité ni de fidélité sémantique. Ne pas utiliser seul pour un conseil engageant des décisions importantes. Pas d'ASR, TTS ou compréhension audio. Le playground éventuel de la plateforme ne prouve pas que ce checkpoint y est servi : aucun backend MandiaCompact public n'est connecté pour l'instant.

## Fichiers
Le fichier `model.pt` d'inférence contient `config`, `state_dict`, `tokenizer_sha256`. Chargement avec `torch.load(..., weights_only=True)`, empreintes obligatoires. Les checkpoints AdamW/RNG restent dans le Drive de travail. La licence Apache-2.0 concerne le code propre ; les attributions et conditions des corpus restent distinctes. Aucune prétention à la première traduction baoulé ou à un résultat état de l'art.

## Variante de recherche étape 7 : question avec contexte

Architecture inchangée ; deux bras de 400 mises à jour, meilleurs poids éligibles au pas 50, bras enrichi retenu sur validation. 40 familles baoulé déjà vues dans le train de l'étape 5, avec questions et reformulations françaises rédigées par l'assistant. Test réservé au niveau des formulations seulement : 6/40 exactes et chrF++ 32,592 contre 4/40 et 29,347 au témoin. La relecture constate encore des erreurs de personne et d'objet ; anciens tests en légère régression. Ce checkpoint est distribué séparément et ne remplace pas l'étape 6 comme preuve de conversation générale. Aucune validation native ni nouveauté architecturale revendiquée.
