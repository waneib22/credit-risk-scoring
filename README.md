# Credit Risk Scoring — Freddie Mac

Modèle de scoring du risque de défaut sur prêts immobiliers, de la donnée brute au déploiement : pipeline reproductible (Kedro), explicabilité (SHAP), application interactive (Streamlit) et assistant documentaire (RAG).

**Stack :** Python · pandas · scikit-learn · XGBoost · SHAP · Kedro · FAISS · Anthropic Claude · Streamlit

---

## Problème

Prédire la **probabilité de défaut** d'un prêt immobilier à partir des seules variables connues **au moment de l'octroi**. L'enjeu métier : aider à décider d'accorder ou non un prêt, et expliquer la décision — une exigence réglementaire (Bâle III, guidelines EBA, droit à l'explication RGPD).

**Donnée :** [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) — Standard Dataset, échantillons 2017 et 2018 (100 000 prêts, performance jusqu'à sept. 2025).
**Cible :** `default = 1` si 90+ jours de retard OU saisie/REO. Taux de défaut observé : **5.57 %**.

---

## Résultats

| Modèle | AUC ROC | Average Precision |
|---|---|---|
| Régression logistique *(retenue en production)* | **0.740** | 0.147 |
| XGBoost *(challenger + analyse SHAP)* | 0.735 | 0.142 |

**Deux enseignements clés :**

1. **« Fancier ≠ better ».** Les deux modèles convergent vers AUC ≈ 0.74 — le plafond de prédictibilité des données d'octroi. Les relations FICO/DTI/LTV → défaut sont quasi-linéaires, donc la régression logistique (interprétable, conforme régulateur) égale XGBoost. Le signal résiduel dépend de chocs macro post-octroi (COVID, taux), inconnus à l'octroi.

2. **L'importance d'une variable n'est pas son utilité prédictive.** SHAP désignait `postal_code` comme driver #1, mais le retirer a *amélioré* l'AUC (0.72 → 0.735) : sa cardinalité trop fine provoquait un overfit géographique. Seule la validation sur données vierges tranche.

**Top drivers (SHAP) :** `fico_dti_interaction`, `msa`, `number_of_borrowers`, `fico_ltv_interaction`, `credit_score`.

---

## Architecture

```
data project/
├── conf/                    # configuration Kedro (catalog, parameters)
│   └── base/
│       ├── catalog.yml       # déclaration des datasets
│       └── parameters.yml    # hyperparamètres
├── data/
│   └── raw/                  # fichiers Freddie Mac (non versionnés)
├── knowledge_base/           # documentation indexée par le RAG (markdown)
├── models/                   # modèles + préprocesseur + index FAISS
├── notebooks/                # exploration et narratif (01 → 06)
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_shap.ipynb
│   └── 06_rag.ipynb
├── src/credit_risk/          # package : pipeline Kedro + code partagé
│   ├── features.py           # reconstruction du vecteur de features (app)
│   ├── rag.py                # logique RAG (notebook 06 + app)
│   └── pipelines/
│       ├── data_processing/  # raw → features → split
│       └── data_science/     # train → évaluation
└── app/app.py                # application Streamlit
```

**Deux chemins complémentaires :**
- **Notebooks** = exploration et narratif pédagogique (EDA, choix de modélisation, SHAP).
- **Pipeline Kedro** = version industrialisée, reproductible, paramétrée. Reproduit exactement les métriques des notebooks.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # installe le package credit_risk
```

Place les fichiers Freddie Mac (`sample_orig_*.txt`, `sample_svcg_*.txt`) dans `data/raw/`.

---

## Utilisation

### 1. Pipeline Kedro (entraînement reproductible)

```bash
kedro run                       # pipeline complet (data_processing + data_science)
kedro run --pipeline data_processing   # uniquement la préparation
kedro registry list             # liste les pipelines
kedro viz                       # visualise le DAG dans le navigateur
```

Produit `data/X_train.parquet` … `models/{preprocessor,logreg,xgb_model}.pkl` et `models/metrics.json`.

### 2. Application Streamlit

```bash
streamlit run app/app.py
```

- **Onglet Scoring** : saisie d'un profil emprunteur → probabilité de défaut + explication SHAP.
- **Onglet Assistant** : chatbot RAG sur la documentation du projet.

### 3. Assistant RAG (génération Claude — optionnel)

Crée un fichier `.env` à la racine :
```
ANTHROPIC_API_KEY=sk-ant-...
```
Sans clé, le RAG fonctionne en mode local (affichage des passages pertinents, sans génération).

---

## Choix méthodologiques

- **Split avant encodage** : le target encoding géographique est calculé sur le train uniquement, pour éviter le data leakage.
- **Retrait des variables de performance** : elles servent à construire la cible (leakage).
- **Gestion du déséquilibre** : `class_weight='balanced'` (LogReg), `eval_metric='aucpr'` (XGBoost).
- **Explicabilité** : SHAP décompose chaque prédiction de façon additive — exploitable pour justifier un refus client.
