# Apprentissage assisté par un grand modèle

Étape 7 effectivement exécutée le 14 octobre 2026 : augmentation de questions et reformulations françaises, cibles baoulé attestées recopiées du TRAIN Wiktionary/Kaikki étape 5. Aucun poids ou logits privés de l'assistant, aucune chaîne de raisonnement privée, aucune nouvelle traduction baoulé inventée. Cette méthode relève de l'augmentation de données assistée, pas d'une démonstration de distillation de connaissances baoulé par un enseignant fiable.

Articles motivants, dont les gains ne sont pas ceux de Mandia :

- Kim & Rush (2016), [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/), apprentissage sur les sorties d'un enseignant en traduction.
- Hsieh et al. (2023), [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301), supervision auxiliaire dans les tâches évaluées par les auteurs.

Deux bras de 400 pas depuis le même parent : canonique ou enrichi. Mêmes identifiants et tokens supervisés, longueurs source différentes et comptées. Sélection sur validation avec garde sur les anciens dialogues, avant ouverture du test de formulation. Les mêmes 40 cibles sont présentes dans les trois splits : ne pas présenter cela comme une généralisation à de nouvelles réponses.

Résultat : français enrichi sélectionné au pas 50 ; 6/40 réponses exactes contre 4/40 au témoin, chrF++ 32,592 contre 29,347. Les deux paires de contextes contrastés ne sont pas entièrement correctes ; les anciens tests régressent légèrement. Les contrôles doivent primer sur l'envie d'annoncer une forte amélioration. Voir les rapports et la relecture technique complète.

Pistes de données : [LGMEF/CNRS Baoulé](https://lgidf.cnrs.fr/baoule), [CIEL-F](https://moca.phil2.uni-freiburg.de/moca3_v3/index.php?vi=12), [Common Voice 27](https://mozilladatacollective.com/datasets/cmu625dl700nxnq07jev059gj), [modèle externe NLLB baoulé](https://huggingface.co/GaindeNdiaye/nllb-baoule-v1). Aucun nouveau grand corpus de dialogues naturels français-baoulé vérifié et réutilisable n'a été obtenu. Common Voice est de la parole lue ; les autres pistes ont des conditions d'accès et de réutilisation à établir avant admission. Les poids NLLB n'ont pas été utilisés et leurs scores ne sont pas comparables à notre protocole.

La suite doit améliorer la fidélité aux personnes/objets et préserver les anciens domaines, tout en évaluant sur de nouvelles cibles et des dialogues réellement attestés. Un simple accroissement de paramètres ou davantage de copies d'exemples ne résout pas à lui seul ce besoin. Aucun résultat de cette étape ne justifie une annonce d'état de l'art ou de maîtrise générale du baoulé.
