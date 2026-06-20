# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) pipeline for analyzing SEC 10-K filings. Given a stock ticker, it downloads filings from SEC EDGAR, cleans the full document text, chunks with section-aware metadata, embeds into a persistent ChromaDB, reranks with a cross-encoder, and answers financial questions using FLAN-T5.

## Running the App

```bash
# Start the Gradio web interface
python app.py

# Use the agent directly from Python
from agent import call_agent
call_agent("AAPL", "What are Apple's major risk factors?")
```

Individual pipeline stages can be run standalone:
```bash
python downloader.py   # interactive prompt for ticker symbol
python extractor.py    # cleans AAPL filings by default (hardcoded)
python vectordb.py     # builds ChromaDB collection from cleaned_filings/
python agent.py        # runs a default AAPL query
```

## Installing Dependencies

No requirements.txt exists. Install manually:
```bash
pip install sec-edgar-downloader beautifulsoup4 chromadb sentence-transformers transformers torch gradio langchain tqdm
```

`generate_report.py` additionally requires `reportlab`.

## Architecture

The pipeline is a linear 5-module chain: **downloader → extractor → vectordb → agent → app**

- **downloader.py** — Uses `sec-edgar-downloader` to fetch up to 5 years of 10-K filings from SEC EDGAR into `sec-edgar-filings/{SYMBOL}/10-K/`. The `Downloader` is instantiated with a hardcoded UT Austin identity and email.
- **extractor.py** — Parses `full-submission.txt` HTML with BeautifulSoup, strips tags, normalizes whitespace. Saves one cleaned text file per filing as `cleaned_filings/{SYMBOL}_{YEAR}.txt`. Year is inferred from the 2-digit number in the SEC accession folder name.
- **vectordb.py** — Loads cleaned text files per company, chunks (1000 chars, 200 overlap) with LangChain's `RecursiveCharacterTextSplitter`, detects ITEM headings to tag each chunk's section, embeds with `all-MiniLM-L6-v2`, and stores in **persistent** ChromaDB (`chroma_db/`). Skips re-indexing if a company is already present. Metadata per chunk: company, year, section, source.
- **agent.py** — Orchestrates the full pipeline. Downloads/cleans filings if not cached, builds the vector collection, retrieves top-20 candidates filtered by company, **reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`** to select top-5, and generates with `google/flan-t5-large` (seq2seq, beam search=4, temperature=0.4, max 1024 input tokens). Models are lazy-loaded and cached.
- **app.py** — Gradio `Blocks` UI with ticker + question inputs and a text output.

**generate_report.py** is a standalone ReportLab script that produces an interview prep PDF. It is not part of the RAG pipeline.

## Generated Directories (not in git)

- `sec-edgar-filings/` — Raw downloaded filings
- `cleaned_filings/` — Full cleaned text per filing (one file per company-year)
- `chroma_db/` — Persistent ChromaDB vector store

## Known Limitations

- FLAN-T5-Large input is truncated to 1024 tokens — long contexts are silently clipped.
- Section detection via ITEM heading regex is best-effort; chunks in non-standard filings may get tagged as the wrong section or "PREAMBLE".
