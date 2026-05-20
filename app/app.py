"""
Credit Risk Scoring — App Streamlit
Deux onglets :
  1. Scoring : saisie d'un profil emprunteur → probabilité de défaut + explication SHAP
  2. Assistant : chatbot RAG sur la documentation du projet

Lancement : streamlit run app/app.py
"""
import os
import pickle
import sys

import pandas as pd
import streamlit as st

# Chemins relatifs à la racine du projet (app/ est un sous-dossier)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

# Code partagé dans le package credit_risk (src/credit_risk/)
sys.path.insert(0, os.path.join(ROOT, "src"))
from credit_risk.features import build_feature_vector
from credit_risk.rag import retrieve as rag_retrieve, answer as rag_answer, has_api_key

st.set_page_config(page_title="Credit Risk Scoring — Freddie Mac", layout="wide")

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


# ---------------------------------------------------------------------------
# Chargement des artefacts (mis en cache pour ne charger qu'une fois)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "xgb_credit_risk.json"))

    with open(os.path.join(MODELS_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "shap_explainer.pkl"), "rb") as f:
        explainer = pickle.load(f)

    return model, metadata, preprocessor, explainer


@st.cache_resource
def load_rag_artifacts():
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(os.path.join(MODELS_DIR, "rag_index.faiss"))
    with open(os.path.join(MODELS_DIR, "rag_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embedder


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏠 Credit Risk Scoring — Freddie Mac")
st.caption("Modèle de prédiction de défaut sur prêts immobiliers · XGBoost + SHAP + RAG")

tab_score, tab_chat = st.tabs(["📊 Scoring", "💬 Assistant"])

# ----------------------------- Onglet Scoring -----------------------------
with tab_score:
    try:
        model, metadata, preprocessor, explainer = load_model_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Artefacts manquants. Exécute les notebooks 03, 04 et 05 pour générer "
            f"les fichiers de `models/`.\n\nDétail : {e}"
        )
        st.stop()

    st.subheader("Profil de l'emprunteur")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Score FICO", 300, 850, 720)
        dti = st.slider("DTI (%)", 0, 65, 35)
        ocltv = st.slider("LTV combiné (%)", 5, 105, 80)
    with c2:
        original_upb = st.number_input("Montant du prêt ($)", 20_000, 1_500_000, 200_000, step=10_000)
        original_interest_rate = st.slider("Taux d'intérêt (%)", 2.0, 8.0, 4.25, step=0.125)
        original_loan_term = st.selectbox("Durée (mois)", [360, 240, 180, 120], index=0)
    with c3:
        property_state = st.selectbox("État", US_STATES, index=US_STATES.index("CA"))
        number_of_borrowers = st.selectbox("Nombre d'emprunteurs", [1, 2, 3, 4], index=1)
        first_time_homebuyer_flag = st.radio("Primo-accédant", ["N", "Y"], horizontal=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        channel = st.selectbox("Canal", ["R", "C", "B"], format_func=lambda x: {"R": "Retail", "C": "Correspondent", "B": "Broker"}[x])
    with c5:
        occupancy_status = st.selectbox("Occupation", ["P", "S", "I"], format_func=lambda x: {"P": "Résidence principale", "S": "Secondaire", "I": "Investissement"}[x])
    with c6:
        loan_purpose = st.selectbox("Motif", ["P", "N", "C"], format_func=lambda x: {"P": "Achat", "N": "Refi sans cash", "C": "Refi cash-out"}[x])
    property_type = st.selectbox("Type de bien", ["SF", "CO", "PU", "MH", "CP"],
                                 format_func=lambda x: {"SF": "Maison individuelle", "CO": "Condo", "PU": "PUD", "MH": "Mobil-home", "CP": "Coopérative"}[x])

    if st.button("Évaluer le risque", type="primary"):
        inp = dict(
            credit_score=credit_score, dti=dti, ocltv=ocltv, original_upb=original_upb,
            original_interest_rate=original_interest_rate, original_loan_term=original_loan_term,
            property_state=property_state, number_of_borrowers=number_of_borrowers,
            first_time_homebuyer_flag=first_time_homebuyer_flag, channel=channel,
            occupancy_status=occupancy_status, loan_purpose=loan_purpose, property_type=property_type,
        )
        X = build_feature_vector(inp, preprocessor)
        proba = float(model.predict_proba(X)[:, 1][0])

        st.subheader("Résultat")
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Probabilité de défaut", f"{proba:.1%}")
            base = metadata.get("default_rate", 0.0557)
            if proba < base:
                st.success("Risque inférieur à la moyenne du portefeuille")
            elif proba < 2 * base:
                st.warning("Risque modéré")
            else:
                st.error("Risque élevé")
            st.caption(f"Moyenne portefeuille : {base:.1%}")

        # Explication SHAP
        with m2:
            st.markdown("**Pourquoi ce score ? (contributions SHAP)**")
            shap_values = explainer(X)
            contribs = pd.DataFrame({
                "feature": X.columns,
                "shap": shap_values.values[0],
            })
            contribs["abs"] = contribs["shap"].abs()
            top = contribs.sort_values("abs", ascending=False).head(8).sort_values("shap")
            st.bar_chart(top.set_index("feature")["shap"], horizontal=True)
            st.caption("Valeurs > 0 (rouge) poussent vers le défaut ; < 0 protègent. "
                       "Référence : log-odds moyen du modèle.")

# ----------------------------- Onglet Assistant -----------------------------
with tab_chat:
    st.subheader("Assistant documentaire (RAG)")
    st.caption("Pose une question sur le modèle, les variables ou la méthodologie.")

    try:
        index, chunks, embedder = load_rag_artifacts()
    except FileNotFoundError as e:
        st.error(f"Index RAG manquant. Exécute le notebook 06.\n\nDétail : {e}")
        st.stop()

    if not has_api_key():
        st.info("Mode local (pas de clé API) : affichage des passages pertinents sans génération Claude. "
                "Ajoute ta clé dans `.env` pour activer les réponses rédigées.")

    question = st.text_input("Ta question", placeholder="Ex : Pourquoi avoir retiré le code postal du modèle ?")
    if question:
        hits = rag_retrieve(question, index, chunks, embedder, k=4)
        answer, context = rag_answer(question, hits)

        if answer:
            st.markdown("### Réponse")
            st.write(answer)
            with st.expander("Sources utilisées"):
                for h in hits:
                    st.markdown(f"**{h['source']} — {h['title']}** (score {h['score']:.2f})")
        else:
            st.markdown("### Passages les plus pertinents")
            for h in hits:
                with st.expander(f"{h['source']} — {h['title']} (score {h['score']:.2f})"):
                    st.write(h["text"])
