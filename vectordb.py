import os
import re
import uuid
from tqdm import tqdm
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

CLEANED_DIR = "./cleaned_filings"
DB_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sec_10k_filings"
BATCH_SIZE = 500

SECTION_PATTERN = re.compile(r"ITEM\s+(\d+[A-Z]?)\b", re.IGNORECASE)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", ".", " ", ""],
)

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def load_text_files(company):
    data = []
    for fname in os.listdir(CLEANED_DIR):
        if not fname.endswith(".txt") or not fname.startswith(company):
            continue
        parts = fname.replace(".txt", "").split("_", 1)
        year = parts[1] if len(parts) > 1 else "unknown"
        with open(os.path.join(CLEANED_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        data.append({"filename": fname, "text": text, "company": company, "year": year})
    return data


def assign_sections(chunks):
    """Tag each chunk with the most recent ITEM heading seen so far."""
    current_section = "PREAMBLE"
    results = []
    for chunk in chunks:
        headings = SECTION_PATTERN.findall(chunk)
        if headings:
            current_section = f"ITEM_{headings[-1].upper()}"
        results.append({"text": chunk, "section": current_section})
    return results


def build_collection(company):
    embedder = get_embedder()

    client = PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing = collection.get(where={"company": company}, limit=1)
    if existing and existing["ids"]:
        print(f"{company} already indexed ({collection.count()} total chunks)")
        return collection, embedder

    docs = load_text_files(company)
    if not docs:
        print(f"No cleaned filings found for {company} in {CLEANED_DIR}/")
        return collection, embedder

    print(f"Found {len(docs)} filings for {company}. Chunking...")

    all_chunks = []
    all_metas = []
    for doc in docs:
        text_chunks = _splitter.split_text(doc["text"])
        sectioned = assign_sections(text_chunks)
        for item in sectioned:
            all_chunks.append(item["text"])
            all_metas.append({
                "company": doc["company"],
                "year": doc["year"],
                "section": item["section"],
                "source": doc["filename"],
            })

    print(f"Embedding {len(all_chunks)} chunks...")
    all_embeddings = embedder.encode(all_chunks, show_progress_bar=True, batch_size=64)

    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Storing in ChromaDB"):
        end = min(i + BATCH_SIZE, len(all_chunks))
        collection.add(
            documents=all_chunks[i:end],
            embeddings=all_embeddings[i:end].tolist(),
            metadatas=all_metas[i:end],
            ids=[str(uuid.uuid4()) for _ in range(end - i)],
        )

    print(f"Stored {collection.count()} total chunks in ChromaDB ({DB_DIR}/)")
    return collection, embedder


if __name__ == "__main__":
    build_collection("AAPL")
