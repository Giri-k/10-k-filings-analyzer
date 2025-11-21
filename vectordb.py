import os
import uuid
from tqdm import tqdm
from chromadb import Client
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

EXTRACTED_DIR = "./sections"     # folder where extractor saved the txt files
DB_DIR = "./chroma_db"                 # folder to persist ChromaDB
CHUNK_SIZE = 1000                      # number of characters per chunk
CHUNK_OVERLAP = 200                    # overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2" # "all-MiniLM-L6-v2" or "BAAI/bge-small-en" text-embedding-3-small
COLLECTION_NAME = "sec_10k_filings"


def load_text_files(folder_path):
    """Load all .txt files from the extracted folder."""
    data = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            data.append({"filename": fname, "text": text})
            #print(data)
    return data


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split a long text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def build_collection():
    print("Loading extracted 10-K text files...")
    docs = load_text_files(EXTRACTED_DIR)
    print(f"Found {len(docs)} files.\n")

    print("Initializing embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Initializing ChromaDB client...")
    client = Client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for doc in tqdm(docs, desc="Processing files"):
        text_chunks = chunk_text(doc["text"])
        embeddings = embedder.encode(text_chunks, show_progress_bar=False)

        # optional: extract metadata like company, year, or section from filename
        metadata = {
            "source": doc["filename"],
            "year": "".join([c for c in doc["filename"] if c.isdigit()])[:4] or "unknown"
        }

        for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )

    print(f"Successfully stored {collection.count()} chunks in ChromaDB!")
    #print(f"Database persisted at: {DB_DIR}")

    #query = "What are Apple's major risk factors mentioned?"
    #results = collection.query(query_texts=[query], n_results=5)

    # for doc in results["documents"][0]:
    #     print("----")
    #     print(doc[:500])
    return collection, embedder


if __name__ == "__main__":
    #main()
    build_collection()
    #load_text_files(EXTRACTED_DIR)
    #verify_chroma_db()