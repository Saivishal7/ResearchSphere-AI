# ingestion/build_vectordb.py
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from config import CHROMA_STORE_DIR
from ingestion.load_faculty import load_all_faculty_profiles
from ingestion.embed_documents import compile_faculty_document, embed_documents

def build_vector_database(directory_path: str = "data/faculty", collection_name: str = "faculty_profiles"):
    """
    Complete RAG ingestion pipeline:
    1. Loads all faculty JSON files.
    2. Compiles detailed semantic documents.
    3. Generates high-quality vector embeddings.
    4. Persists embeddings, documents, and specific metadata inside a local ChromaDB.
    """
    # 1. Load profiles
    print("[Pipeline] Step 1: Loading faculty JSON profiles...")
    profiles = load_all_faculty_profiles(directory_path)
    total_faculty = len(profiles)
    print(f"Total Faculty Loaded: {total_faculty}")
    
    if total_faculty == 0:
        print("[Warning] Ingestion aborted. No valid faculty profiles found.")
        return
        
    # 2. Compile documents and gather metadata
    documents = []
    metadatas = []
    ids = []
    
    for profile in profiles:
        doc = compile_faculty_document(profile)
        documents.append(doc)
        
        # Build metadata dictionary containing the specific requested fields
        metadata = {
            "faculty_id": profile.faculty_id,
            "name": profile.name,
            "department": profile.department,
            "capacity_status": profile.capacity_status
        }
        metadatas.append(metadata)
        ids.append(profile.faculty_id)
        
    # 3. Generate embeddings
    print("Embedding Progress: Generating vector embeddings using sentence-transformers...")
    embeddings = embed_documents(documents)
    print(f"Embedding Progress: Successfully generated {len(embeddings)} embeddings.")
    
    # 4. Initialize and persist ChromaDB
    print(f"[Pipeline] Step 4: Initializing ChromaDB persistent client at '{CHROMA_STORE_DIR}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)
    
    # Use get_or_create_collection for idempotency and safe updates
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Upsert data to avoid key duplicate conflicts if pipeline is executed multiple times
    print("[Pipeline] Indexing data in ChromaDB...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print("Database Created Successfully")

if __name__ == "__main__":
    build_vector_database()
