import json
import os
import numpy as np
import requests
from vectordb import build_collection
from sentence_transformers import CrossEncoder
from downloader import download
from extractor import process_10k_filings, CLEANED_DIR

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def _load_reranker():
    global _reranker
    if _reranker is not None:
        return
    _reranker = CrossEncoder(RERANKER_MODEL)


def retrieve_chunks(query, collection, embedder, bm25_index,
                    all_chunks, all_metas, company,
                    top_k=5, candidates=20):
    query_emb = embedder.encode([query])
    dense_results = collection.query(
        query_embeddings=query_emb,
        n_results=candidates,
        where={"company": company},
    )
    dense_docs = dense_results["documents"][0]
    dense_metas = dense_results["metadatas"][0]

    bm25_scores = bm25_index.get_scores(query.lower().split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:candidates]

    seen = set()
    merged = []
    for doc, meta in zip(dense_docs, dense_metas):
        if doc not in seen:
            seen.add(doc)
            merged.append((doc, meta))
    for idx in bm25_top_idx:
        doc = all_chunks[idx]
        if doc not in seen:
            seen.add(doc)
            merged.append((doc, all_metas[idx]))

    pairs = [[query, doc] for doc, _ in merged]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)

    top_docs = [doc for (doc, _), _ in ranked[:top_k]]
    top_metas = [meta for (_, meta), _ in ranked[:top_k]]
    return top_docs, top_metas


def generate_insights(query, context):
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
    if LLM_BACKEND == "huggingface":
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            model=HF_MODEL, token=os.environ.get("HF_TOKEN")
        )
        full_text = ""
        for chunk in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            yield full_text
    else:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
        }, stream=True)
        response.raise_for_status()
        full_text = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                full_text += data.get("response", "")
                yield full_text


def call_agent(symbol, query):
    """Yields (status, output) tuples for streaming UI updates."""
    symbol = symbol.strip().upper()
    if not symbol or not query.strip():
        yield "Please enter a ticker symbol and question.", ""
        return

    filings_dir = f"./sec-edgar-filings/{symbol}/10-K"

    if not os.path.exists(filings_dir):
        yield "Downloading 10-K filings from SEC EDGAR...", ""
        download(symbol)

    has_cleaned = os.path.exists(CLEANED_DIR) and any(
        f.startswith(symbol) for f in os.listdir(CLEANED_DIR)
    )
    if not has_cleaned:
        yield "Cleaning and extracting text from filings...", ""
        process_10k_filings(symbol)

    yield "Building vector index...", ""
    collection, embedder, bm25_index, chunks, metas = build_collection(symbol)

    yield "Loading reranker model...", ""
    _load_reranker()

    yield "Searching with hybrid retrieval (BM25 + dense)...", ""
    context, meta = retrieve_chunks(
        query, collection, embedder, bm25_index, chunks, metas, symbol
    )

    yield "Generating analysis...", ""
    for partial in generate_insights(query, "\n\n".join(context)):
        yield "Generating analysis...", partial

    yield "Done.", partial


if __name__ == "__main__":
    for status, text in call_agent("AAPL", "What are Apple's major risk factors mentioned?"):
        print(f"[{status}]")
    print(text)
