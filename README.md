---
title: 10-K Filings Analyzer
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: false
---

# 10-K Filings Analyzer

A Retrieval-Augmented Generation (RAG) application for analyzing SEC 10-K filings. Given a stock ticker, the system downloads filings from SEC EDGAR, chunks and embeds the full documents, and answers financial questions using hybrid retrieval, cross-encoder reranking, and LLM-powered generation via Ollama.

## Architecture

The application follows a RAG pipeline with hybrid retrieval:

1. **Download** — Fetches up to 5 years of 10-K filings from SEC EDGAR
2. **Clean** — Strips HTML tags and normalizes the full document text
3. **Chunk** — Splits documents into overlapping chunks with section-aware metadata (ITEM headings detected automatically)
4. **Embed** — Creates vector embeddings using Sentence Transformers
5. **Store** — Persists embeddings and metadata in ChromaDB (survives across runs)
6. **Retrieve (Hybrid)** — Combines BM25 sparse search (keyword matching) with dense cosine similarity to find candidates
7. **Rerank** — Cross-encoder reranker scores all candidates and selects the top results
8. **Generate** — Ollama (Llama 3.1) generates structured financial insights from the retrieved context

## Tech Stack

| Component | Technology |
|---|---|
| Filing download | `sec-edgar-downloader` |
| HTML parsing | BeautifulSoup |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB (persistent) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM generation | Ollama (`llama3.1`) |
| Frontend | Gradio |

## Project Structure

```
10-k-filings-analyzer/
├── app.py              # Gradio web interface
├── agent.py            # RAG orchestration: hybrid retrieval, reranking, generation
├── downloader.py       # Downloads 10-K filings from SEC EDGAR
├── extractor.py        # Cleans and extracts full text from filing HTML
├── vectordb.py         # Chunking, embedding, ChromaDB + BM25 index management
├── sec-edgar-filings/  # Raw downloaded filings (generated)
├── cleaned_filings/    # Full cleaned text per filing (generated)
├── chroma_db/          # Persistent vector database (generated)
└── bm25_cache/         # Cached chunks for BM25 index (generated)
```

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running
- Internet connection (for downloading filings and models on first run)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install sec-edgar-downloader beautifulsoup4 chromadb sentence-transformers langchain rank_bm25 requests numpy gradio tqdm
   ```

2. **Install and start Ollama:**
   ```bash
   # Install from https://ollama.com/
   ollama pull llama3.1
   ollama serve
   ```

## Usage

### Web Interface

```bash
python app.py
```

Enter a company ticker (e.g., `AAPL`, `MSFT`, `GOOGL`) and a question. The first run for a ticker will download and process filings automatically; subsequent queries use cached data.

### Command Line

```bash
python agent.py
```

### Python API

```python
from agent import call_agent

insights = call_agent("AAPL", "What are Apple's major risk factors?")
print(insights)
```

### Example Queries

- "What are Apple's major risk factors mentioned?"
- "What does management discuss about future outlook?"
- "What are the key financial trends in the recent filings?"
- "What does the company say about supply chain risks?"

## How It Works

### Retrieval Pipeline

The system uses a three-stage retrieval pipeline for high-quality context selection:

1. **Hybrid candidate retrieval** — Retrieves top-20 candidates from both BM25 (keyword matching) and ChromaDB (semantic similarity), then merges and deduplicates. BM25 catches exact keyword matches that dense retrieval misses; dense retrieval handles paraphrased or conceptual queries.

2. **Cross-encoder reranking** — All merged candidates are scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) which reads the full query-document pair jointly. The top 5 are selected for generation.

3. **LLM generation** — The selected chunks are passed as context to Llama 3.1 via Ollama, which produces a structured analysis with key points, observations, and red flags.

### Section-Aware Chunking

Instead of extracting specific sections via regex (fragile across different filing formats), the system chunks the entire document and tags each chunk with the nearest preceding ITEM heading (e.g., `ITEM_1A`, `ITEM_7`). This metadata is stored in ChromaDB and can be used for filtering.

### Persistence

- **ChromaDB** persists embeddings to disk — re-indexing is skipped if a company is already stored
- **BM25 cache** saves chunked text as pickle files — BM25 index rebuilds in <1 second from cache
- **Cleaned filings** are saved as text files — HTML parsing is only done once per filing

## Configuration

| Setting | File | Default |
|---|---|---|
| Ollama model | `agent.py` → `OLLAMA_MODEL` | `llama3.1` |
| Reranker model | `agent.py` → `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Embedding model | `vectordb.py` → `EMBED_MODEL` | `all-MiniLM-L6-v2` |
| Chunk size | `vectordb.py` → `CHUNK_SIZE` | 1000 characters |
| Chunk overlap | `vectordb.py` → `CHUNK_OVERLAP` | 200 characters |
| Retrieval candidates | `agent.py` → `retrieve_chunks()` | 20 per source (BM25 + dense) |
| Final top-k | `agent.py` → `retrieve_chunks()` | 5 (after reranking) |
| Filing years | `downloader.py` → `dl.get()` | 5 most recent |

## Troubleshooting

### Ollama connection error
Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.1`).

### Empty or missing filings
Some SEC accession folders may not contain `full-submission.txt`. The extractor skips these and logs a message. Re-running the downloader may help.

### Slow first run
The first query for a new ticker downloads filings, parses HTML, embeds ~40K chunks, and builds the BM25 index. Subsequent queries for the same ticker skip all of this.

## License

This project is for educational and research purposes.
