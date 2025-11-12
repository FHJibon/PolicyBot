import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_core._api.deprecation')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import config
from pinecone import Pinecone
from src.create_index import create_pinecone_index
from src.data_loader import load_and_chunk_pdfs
from src.vector_store import get_embedding_model
from langchain_community.vectorstores import Pinecone as PineconeStore

def main_ingest():
    print("Starting ingestion process...")
    create_pinecone_index()
    chunks = load_and_chunk_pdfs(config.DATA_DIR)
    if not chunks:
        print("No chunks to process.")
        return

    print(f"Loaded {len(chunks)} chunks")
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}...")
    
    embeddings = get_embedding_model()
    
    print(f"Uploading {len(chunks)} chunks to Pinecone index '{config.PINECONE_INDEX_NAME}'...")

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_store = PineconeStore.from_documents(
            batch, 
            embeddings, 
            index_name=config.PINECONE_INDEX_NAME
        )
        print(f"Uploaded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    
    print("\n--- Ingestion Complete ---")
    print(f"Successfully uploaded {len(chunks)} chunks to Pinecone.")

if __name__ == "__main__":
    main_ingest()