from sre_parse import Tokenizer
from vectordb import build_collection
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from downloader import download
from extractor import process_10k_filings
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-large"     # or "facebook/bart-large-cnn"

def retrieve_chunks(query, collection, embedder, top_k=5):
    query_emb = embedder.encode([query])
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas

def generate_insights(query, context, model, tokenizer, max_len=512):
    """Use a Hugging Face model to summarize or reason over the retrieved context."""
    prompt = f"""
    You are a financial analyst.
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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_len,
            num_beams=4,
            temperature=0.4,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def call_agent(symbol, query):

    filings_dir = f"./sec-edgar-filings/{symbol}/10-K"
    sections_dir = "./sections"

    if not os.path.exists(filings_dir):
        print("downloading 10-K filings...")
        download(symbol)
        process_10k_filings(symbol)
    else:
        print("10-K filings already downloaded")
    
    collection, embedder = build_collection()  # load or rebuild your DB
    
    global _model, _tokenizer
    if "_model" not in globals() or "_tokenizer" not in globals():
        print("Loading Hugging Face model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    else:
        print("Using cached Hugging Face model...")
    
    print("retrieving chunks...")
    context, metas = retrieve_chunks(query, collection, embedder)
    print("chunks retrieved")

    insights = generate_insights(query, context, _model, _tokenizer)
    print(insights)
    return insights

if __name__ == "__main__":
    call_agent("AAPL", "What are Apple's major risk factors mentioned?")