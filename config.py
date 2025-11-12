import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "Data"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
METRIC = "cosine"
PINECONE_INDEX_NAME = "hr-policy"

TOP_K = 4
LLM_MODEL_NAME = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1