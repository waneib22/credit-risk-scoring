"""Logique RAG partagée entre le notebook 06 et l'app Streamlit.

- chunk_markdown : découpe un markdown en chunks par section.
- retrieve       : recherche les chunks les plus proches dans l'index FAISS.
- answer         : génère une réponse avec Claude à partir du contexte récupéré
                   (ou retourne None en mode local si pas de clé API).
"""
from __future__ import annotations

import os

_MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = (
    "Tu es l'assistant documentaire d'un projet de scoring crédit construit sur le "
    "Freddie Mac Single-Family Loan-Level Dataset. Tu réponds aux questions sur le modèle, "
    "les variables et la méthodologie.\n\n"
    "Règles :\n"
    "- Réponds UNIQUEMENT à partir du CONTEXTE fourni dans le message utilisateur.\n"
    "- Si le contexte ne contient pas la réponse, dis-le clairement plutôt que d'inventer.\n"
    "- Sois concis et précis. Cite les chiffres exacts quand ils sont disponibles.\n"
    "- Réponds en français."
)


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Découpe un markdown en chunks par section de niveau ## ou ###."""
    chunks = []
    current_title = source
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append({
                "source": source,
                "title": current_title,
                "text": f"[{source} — {current_title}]\n{body}",
            })

    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("### "):
            flush()
            current_title = line.lstrip("# ").strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return chunks


def has_api_key() -> bool:
    """True si une clé API Anthropic valide est présente (pas le placeholder)."""
    raw = os.getenv("ANTHROPIC_API_KEY", "")
    return raw.startswith("sk-ant-") and "remplace" not in raw


def retrieve(question: str, index, chunks: list[dict], embedder, k: int = 4) -> list[dict]:
    """Retourne les k chunks les plus proches de la question (similarité cosinus)."""
    import numpy as np

    q_emb = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def answer(question: str, hits: list[dict]):
    """Génère une réponse Claude à partir des chunks récupérés.

    Returns:
        (texte, contexte) si une clé API est présente ;
        (None, contexte) en mode local (pas de génération).
    """
    context = "\n\n---\n\n".join(h["text"] for h in hits)

    if not has_api_key():
        return None, context

    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # préfixe stable mis en cache
        }],
        messages=[{
            "role": "user",
            "content": f"CONTEXTE :\n{context}\n\nQUESTION : {question}",
        }],
    )
    return next(b.text for b in resp.content if b.type == "text"), context
