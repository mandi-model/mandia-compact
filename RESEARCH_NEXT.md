# Prochaine expérience — supervision assistée, non encore entraînée

Les articles suivants motivent une expérience ; leurs gains ne sont pas des résultats Mandia :

- Kim & Rush (2016), Sequence-Level Knowledge Distillation : https://aclanthology.org/D16-1139/ ; apprentissage sur des séquences d'un enseignant en traduction.
- Hsieh et al. (2023), Distilling Step-by-Step! : https://arxiv.org/abs/2305.02301 ; supervision additionnelle dérivée d'un grand modèle sur les tâches évaluées par les auteurs.

Ici, l'assistant peut produire des reformulations françaises, des intentions explicites et des questions reliées à une réponse attestée. Il ne fournit ni poids, ni logits, ni raisonnement privé, et sa connaissance du baoulé ne remplace pas un locuteur.

Protocole proposé : deux continuations du même parent, même nombre de mises à jour et mêmes identifiants/cibles tirés ; témoin texte français canonique, bras enrichi avec reformulations françaises contrôlées. Préserver négation, personne, temps, entités et intention. Les entrées modifiées ont des longueurs différentes : mesurer séparément tokens réels, temps et mémoire, sans revendiquer un budget identique sans preuve. Pas d'augmentation de paramètres pour ce premier contrôle.

Réserver les familles sémantiques avant rédaction ; ne jamais entraîner les cibles des holdouts antérieurs. Séparer explicitement test de reformulations de cibles déjà vues et généralisation à de nouvelles cibles. Sélection sur validation, garde de régression, test final après sélection gelée. Rapporter fidélité aux faits, changement de sujet, répétition et réponses génériques, en plus de chrF++/CE. Sauvegarder AdamW/RNG/compteurs et les empreintes du code/données dans Drive tous les 100 pas.

Recherche du 14 octobre 2026 : https://huggingface.co/GaindeNdiaye/nllb-baoule-v1 signale aussi peu de conversations naturelles et aucune évaluation native. Son NLLB de 600 M n'est ni intégré ni comparé à Mandia : données et protocole différents ; la licence des poids ne remplace pas celle de ses sources. Common Voice 27 est de la parole lue, pas une collection de conversations français-baoulé. Les sources bibliques connues sont déjà indexées ; ne pas mélanger audio 1974 et texte 1998.
