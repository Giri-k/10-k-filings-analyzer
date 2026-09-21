"""
RAGAS evaluation for the 10-K filing RAG pipeline.

Measures four reference-free dimensions using LLM-as-judge (via Ollama):
  - Faithfulness: Is the answer grounded in the retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are the retrieved chunks relevant to the question?
  - Context Utilization: Does the answer use the provided context well?

Usage:
  python evaluate.py              # run eval on AAPL (must be indexed already)
  python evaluate.py --ticker MSFT
"""

import argparse
import json
import os
import warnings
import requests
import numpy as np
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecisionWithoutReference,
    ContextUtilization,
    Faithfulness,
    AnswerRelevancy,
)
from ragas.embeddings import HuggingFaceEmbeddings
from agent import _ensure_indexed, _load_reranker, retrieve_chunks

warnings.filterwarnings("ignore", category=DeprecationWarning)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
EVAL_QUERIES = [
    "What are the company's major risk factors?",
    "What does management discuss about revenue trends?",
    "What are the key supply chain risks?",
    "What does the company say about competition?",
    "What are the main regulatory risks?",
    "What does the company report about research and development?",
    "What are the significant legal proceedings?",
    "What does management say about future outlook?",
    "What are the cybersecurity and data privacy risks?",
    "What are the company's debt and liquidity positions?",
]


def generate_answer_grounded(query, context_docs):
    numbered_sources = []
    for i, doc in enumerate(context_docs, 1):
        numbered_sources.append(f"[Source {i}]\n{doc}")
    context = "\n\n---\n\n".join(numbered_sources)

    prompt = (
        "You are a financial analyst. Based ONLY on the following 10-K excerpts, "
        "answer the question below.\n\n"
        f"---CONTEXT---\n{context}\n\n"
        f"---QUESTION---\n{query}\n\n"
        "RULES:\n"
        "- Base your answer ONLY on the context above. Do NOT add facts from your own knowledge.\n"
        "- Use bullet points. Each point should reference a specific source.\n"
        "- If the context does not contain enough information, say what is missing.\n"
    )
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    })
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def run_evaluation(ticker):
    print(f"=== RAGAS Evaluation for {ticker} ===\n")

    collection, embedder, bm25_index, chunks, metas = _ensure_indexed(ticker)
    _load_reranker()

    client = AsyncOpenAI(base_url=f"{OLLAMA_BASE}/v1", api_key="ollama")
    llm = llm_factory(OLLAMA_MODEL, client=client)
    emb = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "context_utilization": ContextUtilization(llm=llm),
    }

    results = []

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"[{i}/{len(EVAL_QUERIES)}] {query}")

        docs, doc_metas = retrieve_chunks(
            query, collection, embedder, bm25_index, chunks, metas, ticker
        )
        answer = generate_answer_grounded(query, docs)

        row = {"query": query, "answer_preview": answer[:100]}

        for name, metric in metrics.items():
            try:
                kwargs = {"user_input": query, "response": answer}
                if name != "answer_relevancy":
                    kwargs["retrieved_contexts"] = docs
                score = metric.score(**kwargs)
                row[name] = float(score)
            except Exception as e:
                print(f"  Warning: {name} failed — {e}")
                row[name] = None

        def _fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"

        print(f"  F={_fmt(row.get('faithfulness'))}  "
              f"AR={_fmt(row.get('answer_relevancy'))}  "
              f"CP={_fmt(row.get('context_precision'))}  "
              f"CU={_fmt(row.get('context_utilization'))}")

        results.append(row)

    valid_scores = {name: [] for name in metrics}
    for row in results:
        for name in metrics:
            if row.get(name) is not None:
                valid_scores[name].append(row[name])

    print("\n" + "=" * 55)
    print(f"  RAGAS Evaluation Results — {ticker}")
    print("=" * 55)
    for name in metrics:
        scores = valid_scores[name]
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            print(f"  {name:30s}: {avg:.4f} (±{std:.4f})")
        else:
            print(f"  {name:30s}: N/A")
    print("=" * 55)

    out_path = f"eval_results_{ticker}.json"
    with open(out_path, "w") as f:
        json.dump({
            "ticker": ticker,
            "model": OLLAMA_MODEL,
            "num_queries": len(EVAL_QUERIES),
            "averages": {
                name: float(np.mean(scores)) if scores else None
                for name, scores in valid_scores.items()
            },
            "per_query": results,
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for 10-K RAG pipeline")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to evaluate")
    args = parser.parse_args()

    run_evaluation(args.ticker)
