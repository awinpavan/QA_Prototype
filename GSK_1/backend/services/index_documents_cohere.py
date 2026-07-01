# index_documents_cohere.py
import os
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
import cohere # Import cohere
from dotenv import load_dotenv
import re
import time

# --- Initialization ---
# Load environment variables from .env file
load_dotenv()

# CHANGED: Initialize Cohere client
try:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)
except Exception as e:
    print(f"Error initializing Cohere client: {e}")
    exit()

# Initialize Pinecone client (new SDK)
try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit()

# --- Configuration ---
SOP_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sops"))
PINECONE_INDEX_NAME = "gsk-qa-copilot"
# CHANGED: Use a Cohere embedding model
EMBEDDING_MODEL = "embed-english-v3.0"

# Known output dimensions for common Cohere embed models
MODEL_DIMENSIONS = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
}

# --- Functions ---

# NEW: Function to chunk documents into smaller pieces
def chunk_text(text, paragraph_separator="\n\n"):
    """Splits text into paragraphs."""
    chunks = text.split(paragraph_separator)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# CHANGED: Function to get embeddings from Cohere
def get_cohere_embeddings(texts):
    """Generates embeddings for a list of texts using Cohere."""
    try:
        response = co.embed(
            texts=texts,
            model=EMBEDDING_MODEL,
            input_type="search_document" # Use 'search_document' for indexing
        )
        return response.embeddings
    except Exception as e:
        print(f"Error getting embeddings from Cohere: {e}")
        return []

# --- Main script execution ---
if __name__ == "__main__":
    # Ensure the Pinecone index exists (create if missing)
    try:
        if not pc.has_index(PINECONE_INDEX_NAME):
            print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating it...")
            embed_dim = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 1024)
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Optional: brief wait to allow control plane to register the index host
            time.sleep(2)
        else:
            print(f"Index '{PINECONE_INDEX_NAME}' exists.")
    except Exception as e:
        print(f"Error ensuring Pinecone index: {e}")
        exit()

    print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
    # Use name-based connection; now that it exists, this will resolve the host
    index = pc.Index(PINECONE_INDEX_NAME)

    print(f"Reading and chunking documents from '{SOP_DIRECTORY}'...")
    for filename in os.listdir(SOP_DIRECTORY):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(SOP_DIRECTORY, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()

            # 1. Chunk the document
            chunks = chunk_text(doc_text)
            print(f"  - Found {len(chunks)} chunks in {filename}")

            # 2. Get embeddings for all chunks in the document
            chunk_embeddings = get_cohere_embeddings(chunks)

            if not chunk_embeddings:
                print(f"  - Skipping {filename} due to embedding error.")
                continue

            # 3. Prepare vectors for Pinecone upsert
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                # Create a unique ID for each chunk
                chunk_id = f"{filename}-chunk-{i}"
                
                # Create metadata to store the original text and source
                metadata = {
                    "text": chunk,
                    "source_document": filename
                }
                
                vectors_to_upsert.append((chunk_id, chunk_embeddings[i], metadata))

            # 4. Upsert all chunks for the document in one go
            print(f"  - Upserting {len(vectors_to_upsert)} vectors for {filename}...")
            index.upsert(vectors=vectors_to_upsert)

    print("\n✅ Indexing complete!")
    print(f"Index stats: {index.describe_index_stats()}")