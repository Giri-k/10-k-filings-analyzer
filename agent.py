from vectordb import build_collection
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder
import torch
from downloader import download
from extractor import process_10k_filings, CLEANED_DIR
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-large"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_tokenizer = None
_reranker = None


def _load_models():
    global _model, _tokenizer, _reranker
    if _model is not None:
        return
    print("Loading generation model and reranker...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    _reranker = CrossEncoder(RERANKER_MODEL)


def retrieve_chunks(query, collection, embedder, company, top_k=5, candidates=20):
    query_emb = embedder.encode([query])
    results = collection.query(
        query_embeddings=query_emb,
        n_results=candidates,
        where={"company": company},
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    pairs = [[query, doc] for doc in docs]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)

    top_docs = [doc for doc, _, _ in ranked[:top_k]]
    top_metas = [meta for _, meta, _ in ranked[:top_k]]
    return top_docs, top_metas


def generate_insights(query, context, max_len=512):
    prompt = f"""You are a financial analyst.
Based on the following 10-K excerpts, answer the question below
and highlight key risks, trends, or insights.

---CONTEXT---
{context}

---QUESTION---
{query}

Provide a structured and concise summary:
- Key Points
- Observations
- Any Red Flags or Year-to-Year Changes
"""
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_len,
            num_beams=4,
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def call_agent(symbol, query):
    filings_dir = f"./sec-edgar-filings/{symbol}/10-K"

    if not os.path.exists(filings_dir):
        print("Downloading 10-K filings...")
        download(symbol)

    has_cleaned = os.path.exists(CLEANED_DIR) and any(
        f.startswith(symbol) for f in os.listdir(CLEANED_DIR)
    )
    if not has_cleaned:
        print("Cleaning filings...")
        process_10k_filings(symbol)

    collection, embedder = build_collection(symbol)
    _load_models()

    print("Retrieving and reranking chunks...")
    context, metas = retrieve_chunks(query, collection, embedder, symbol)

    insights = generate_insights(query, "\n\n".join(context))
    print(insights)
    return insights


if __name__ == "__main__":
    call_agent("AAPL", "What are Apple's major risk factors mentioned?")
