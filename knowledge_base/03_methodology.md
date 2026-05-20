# Méthodologie — Pipeline du projet

Ce document décrit les étapes du pipeline, l'ordre des notebooks, et les choix méthodologiques.

## Structure du projet
- notebooks/01_data_loading.ipynb — chargement et parsing des fichiers bruts.
- notebooks/02_eda.ipynb — analyse exploratoire.
- notebooks/03_preprocessing.ipynb — feature engineering et encodage.
- notebooks/04_modeling.ipynb — entraînement et évaluation.
- notebooks/05_shap.ipynb — explicabilité.
- notebooks/06_rag.ipynb — chatbot RAG (ce notebook).

## Étape 1 — Chargement des données
Les fichiers Freddie Mac sont au format texte, séparateur pipe (|), sans en-tête, encodage latin-1. Deux fichiers par millésime : origination (sample_orig_YYYY.txt, 32 colonnes) et performance (sample_svcg_YYYY.txt, 32 colonnes). La jointure se fait sur loan_sequence_number. Un piège rencontré : pandas avec un nombre de noms de colonnes inférieur au nombre réel décale silencieusement l'alignement ; il a fallu mapper les 32 colonnes exactes du Standard Dataset.

## Étape 2 — Construction de la cible
Les données de performance (une ligne par prêt par mois) sont agrégées en une ligne par prêt : delinquance maximale, nombre de mois en retard, code de solde final. La cible default est dérivée de ces agrégats. Les variables de performance sont ensuite retirées des features car elles constituent du data leakage (elles servent à définir la cible et ne sont pas connues à l'octroi).

## Étape 3 — Analyse exploratoire (EDA)
On vérifie les distributions, les valeurs manquantes, les taux de défaut par variable. Le binning révèle les relations : FICO décroissant monotone avec le défaut, DTI croissant, LTV avec un saut au-delà de 95%. Les corrélations identifient la redondance oltv/ocltv (0.99).

## Étape 4 — Feature engineering
Création de flags de risque (is_subprime, is_high_ltv, is_high_dti), d'un risk_count additif, d'interactions (fico_dti, fico_ltv, dti_ltv), d'une mensualité approximée et d'un rate_spread. Ces features encodent du savoir métier crédit et aident à la fois la régression logistique (linéarisation) et XGBoost (interactions explicites).

## Étape 5 — Split et encodage
Le split train/test (80/20 stratifié) est fait AVANT le target encoding. C'est crucial : encoder les variables géographiques sur le dataset complet ferait fuiter l'information du test vers le train (data leakage), surestimant l'AUC. Le target encoding est donc calculé uniquement sur le train, avec un lissage qui ramène les modalités rares vers la moyenne globale.

## Étape 6 — Modélisation
Régression logistique (baseline interprétable) vs XGBoost (challenger). Métriques : AUC ROC, Average Precision (AP), courbes ROC et Precision-Recall, matrice de confusion à seuil F1 optimal. Le seuil de décision en production dépendrait du coût asymétrique : un défaut raté (perte du capital) coûte bien plus cher qu'un bon client refusé (manque à gagner).

## Étape 7 — Explicabilité SHAP
TreeExplainer appliqué à XGBoost. Visualisations : beeswarm (impact global et sens), bar (importance moyenne), dependence plots (seuils et non-linéarités), waterfall (décomposition d'une prédiction individuelle).

## Étape 8 — RAG
Ce chatbot indexe la documentation du projet (data dictionary, model card, méthodologie) avec des embeddings sentence-transformers et un index FAISS, puis génère des réponses avec Claude. Il répond aux questions sur le modèle, les variables et la méthodologie.

## Environnement technique
Développement en local sous VS Code, exécution via l'extension. Stack Python : pandas, numpy, scikit-learn, xgboost, shap, sentence-transformers, faiss, anthropic, streamlit. Les données brutes ne sont pas versionnées (gitignore).

## Choix méthodologiques clés à retenir
1. Split avant encodage pour éviter le leakage.
2. Retrait des variables de performance (leakage de la cible).
3. Retrait de postal_code (overfit géographique, valide par la hausse d'AUC).
4. Régression logistique retenue en production pour l'interprétabilité réglementaire.
5. XGBoost et SHAP pour l'analyse et l'explicabilité.
