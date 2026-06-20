import json
import os
import re
import numpy as np
import requests
from vectordb import build_collection
from sentence_transformers import CrossEncoder
from downloader import download
from extractor import process_10k_filings, CLEANED_DIR

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_STEPS = 6

_reranker = None
_index_cache = {}

SYSTEM_PROMPT = """You are a financial analyst assistant with access to SEC 10-K annual filings.

You have the following tool:

search(ticker, query) - Search a company's 10-K filings for information. Filings are downloaded and indexed automatically. Returns relevant excerpts from the most recent 5 years.
  - ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
  - query: What to search for

To use the tool, respond in this exact format:

Thought: <your reasoning>
Action: search
Action Input: {"ticker": "<TICKER>", "query": "<search query>"}

When you have enough information, respond with:

Thought: <your reasoning>
Final Answer: <your structured answer with headers, bullet points, and insights>

Rules:
- Always search before answering. Never make up financial information.
- For comparison questions, search each company separately.
- You can call the tool multiple times.
- Keep your Thought concise (1-2 sentences).
"""


def _load_reranker():
    global _reranker
    if _reranker is not None:
        return
    _reranker = CrossEncoder(RERANKER_MODEL)


def _call_llm(messages):
    if LLM_BACKEND == "huggingface":
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            model=HF_MODEL, token=os.environ.get("HF_TOKEN")
        )
        resp = client.chat_completion(messages=messages, max_tokens=1024)
        return resp.choices[0].message.content
    else:
        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        })
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _call_llm_stream(messages):
    if LLM_BACKEND == "huggingface":
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            model=HF_MODEL, token=os.environ.get("HF_TOKEN")
        )
        full = ""
        for chunk in client.chat_completion(
            messages=messages, max_tokens=2048, stream=True
        ):
            full += chunk.choices[0].delta.content or ""
            yield full
    else:
        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
        }, stream=True)
        resp.raise_for_status()
        full = ""
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                full += data.get("message", {}).get("content", "")
                yield full


def _ensure_indexed(ticker):
    if ticker in _index_cache:
        return _index_cache[ticker]

    filings_dir = f"./sec-edgar-filings/{ticker}/10-K"
    if not os.path.exists(filings_dir):
        download(ticker)

    has_cleaned = os.path.exists(CLEANED_DIR) and any(
        f.startswith(ticker) for f in os.listdir(CLEANED_DIR)
    )
    if not has_cleaned:
        process_10k_filings(ticker)

    result = build_collection(ticker)
    _index_cache[ticker] = result
    return result


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


def _execute_search(ticker, search_query):
    ticker = ticker.strip().upper()
    collection, embedder, bm25_index, chunks, metas = _ensure_indexed(ticker)

    if bm25_index is None:
        return f"No filings found for {ticker}."

    _load_reranker()
    docs, doc_metas = retrieve_chunks(
        search_query, collection, embedder, bm25_index, chunks, metas, ticker
    )

    results = []
    for doc, meta in zip(docs, doc_metas):
        header = f"[{meta.get('company', ticker)} {meta.get('year', '?')} - {meta.get('section', '?')}]"
        results.append(f"{header}\n{doc[:600]}")

    return "\n\n---\n\n".join(results) if results else f"No relevant results found for {ticker}."


def _parse_action(text):
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if action_match and input_match:
        try:
            return action_match.group(1), json.loads(input_match.group(1))
        except json.JSONDecodeError:
            pass
    return None, None


def _extract_final_answer(text):
    match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text


def call_agent(symbol, query):
    """ReAct agent loop. Yields (status, output) tuples."""
    symbol = symbol.strip().upper() if symbol else ""
    if not query.strip():
        yield "Please enter a question.", ""
        return

    yield "Initializing agent...", ""
    _load_reranker()

    user_msg = query
    if symbol:
        user_msg += f"\n\nContext: The user is asking about {symbol}."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    output = ""

    for step in range(MAX_STEPS):
        yield f"Agent thinking (step {step + 1})...", output

        response = _call_llm(messages)

        if "Final Answer:" in response:
            thought = re.search(r"Thought:\s*(.*?)(?=Final Answer:)", response, re.DOTALL)
            if thought:
                output += f"**Thought:** {thought.group(1).strip()}\n\n"
            output += _extract_final_answer(response)
            yield "Done.", output
            return

        action, action_input = _parse_action(response)

        if action is None:
            output += response
            yield "Done.", output
            return

        thought = re.search(r"Thought:\s*(.*?)(?=Action:)", response, re.DOTALL)
        if thought:
            output += f"**Thought:** {thought.group(1).strip()}\n\n"

        ticker = action_input.get("ticker", symbol)
        search_query = action_input.get("query", query)
        output += f"**Searching** {ticker} filings for: *{search_query}*\n\n"
        yield f"Searching {ticker} filings...", output

        observation = _execute_search(ticker, search_query)

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    messages.append({
        "role": "user",
        "content": "Provide your Final Answer now based on all the information gathered.",
    })

    yield "Generating final answer...", output
    output += "---\n\n"
    for partial in _call_llm_stream(messages):
        yield "Generating final answer...", output + partial

    yield "Done.", output + partial


if __name__ == "__main__":
    for status, text in call_agent("AAPL", "What are Apple's major risk factors mentioned?"):
        print(f"[{status}]")
    print(text)
