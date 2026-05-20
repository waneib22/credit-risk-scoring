# Model Card — Modèle de Scoring Crédit Freddie Mac

## Objectif
Prédire la probabilité de défaut d'un prêt immobilier à partir des seules variables connues au moment de l'octroi (origination). Le modèle aide à évaluer le risque crédit avant décision d'octroi.

## Données d'entraînement
- Source : Freddie Mac Single-Family Loan-Level Dataset, Standard Dataset, échantillons 2017 et 2018 (50 000 prêts chacun).
- Volume : 100 000 prêts, joints à leur historique de performance mensuelle.
- Période de performance : jusqu'à septembre 2025, couvrant le choc COVID 2020–2021.
- Taux de défaut observé : 5.57%.
- Split : 80% entraînement / 20% test, stratifié sur la cible. L'entraînement est lui-même re-splité en 64K train / 16K validation pour l'early stopping de XGBoost.

## Modèles testés

### Régression logistique (modèle retenu pour la production)
- AUC ROC sur le test : 0.74.
- Avantages : interprétable, conforme aux exigences réglementaires (Bâle III, guidelines EBA), rapide.
- C'est le modèle recommandé pour un déploiement réel en banque.

### XGBoost (modèle d'analyse)
- AUC ROC sur le test : 0.735.
- Hyperparamètres : max_depth=4, learning_rate=0.05, n_estimators jusqu'à 1000 avec early stopping, scale_pos_weight=1, eval_metric=aucpr.
- Sert principalement à l'analyse d'importance des variables via SHAP.

## Verdict de performance
Les deux modèles convergent vers AUC ≈ 0.74, ce qui correspond au plafond de prédictibilité des données d'origination. La régression logistique égale XGBoost car les relations FICO/DTI/LTV → défaut sont quasi-linéaires monotones. Le signal résiduel non capturé dépend de chocs macroéconomiques postérieurs à l'octroi (COVID, taux directeurs, perte d'emploi), par définition inconnus au moment du prêt. Un AUC de 0.74 est dans la fourchette professionnelle pour le credit scoring (la littérature situe ces modèles entre 0.70 et 0.85).

## Drivers du risque (analyse SHAP)
Par importance décroissante :
1. fico_dti_interaction — profils cumulant FICO faible et DTI élevé.
2. msa — signal géographique (zone métropolitaine).
3. number_of_borrowers — 1 emprunteur = risque, 2 = protecteur.
4. fico_ltv_interaction — FICO faible et fort endettement sur le bien.
5. credit_score — le FICO brut.

## Gestion du déséquilibre de classes
Avec 5.57% de défaut, le dataset est déséquilibré. La régression logistique utilise class_weight='balanced'. XGBoost a finalement neutralisé scale_pos_weight (mis à 1) car sur-pondérer la classe minoritaire dégradait l'AUC ; on privilégie eval_metric='aucpr' à la place.

## Décision de feature engineering importante
La variable postal_code (code postal) a été retirée du modèle. Bien qu'identifiée comme driver #1 par SHAP dans une version antérieure, la retirer a fait *monter* l'AUC (de 0.72 à 0.735 sur XGBoost). Explication : sa cardinalité très élevée (milliers de zones, beaucoup avec moins de 10 prêts) faisait que le target encoding mémorisait le train sans généraliser — un overfit géographique. msa et property_state, plus grossiers, sont plus stables et ont été conservés. Leçon : l'importance d'une feature (SHAP) n'égale pas son utilité prédictive ; seule la validation sur données vierges tranche.

## Encodage des variables
- Variables géographiques haute cardinalité (property_state, msa) : target encoding lissé (smoothing alpha=10), calculé uniquement sur le train pour éviter le data leakage.
- Variables catégorielles basse cardinalité (channel, occupancy_status, property_type, loan_purpose, first_time_homebuyer_flag) : one-hot encoding avec drop_first=True.
- Variables numériques manquantes : imputation par la médiane.

## Limites et précautions
- Le modèle ne capture pas les chocs macroéconomiques postérieurs à l'octroi.
- Le target encoding géographique doit être revalidé dans le temps (stabilité des taux de défaut par zone).
- Pour une nouvelle zone géographique sans historique, l'encodage retombe sur la moyenne globale.
- Le modèle est entraîné sur des prêts conformes Freddie Mac ; il ne couvre pas les prêts jumbo ou non-conformes.

## Explicabilité
Le modèle XGBoost est accompagné d'explications SHAP : pour chaque prêt, la prédiction se décompose de façon additive (valeur de base + contributions par variable). Cela permet d'expliquer un refus ligne par ligne au client, conformément au droit à l'explication (RGPD) et aux guidelines EBA sur l'IA en crédit.
