import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_core._api.deprecation')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*langchain.*deprecated.*')

from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import config 

_embedding_model = None
_vector_store_instance = None
_llm_instance = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  
        )
        print("Embedding model loaded and cached")
    return _embedding_model

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            api_key=os.environ.get("OPENAI_API_KEY"),
            streaming=False,
            max_retries=2,
            timeout=30
        )
        print("LLM loaded and cached")
    return _llm_instance

def get_vector_store():
    global _vector_store_instance
    if _vector_store_instance is None:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name = config.PINECONE_INDEX_NAME
        pinecone_index = pc.Index(index_name)
        
        embeddings = get_embedding_model()
        
        _vector_store_instance = PineconeStore(pinecone_index, embeddings, "text")
        print("Vector store initialized")
    return _vector_store_instance

def query_vector_store(query_text):
    vector_store = get_vector_store()
    relevant_docs = vector_store.similarity_search(
        query_text, 
        k=config.TOP_K
    )
    return relevant_docs