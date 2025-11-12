import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import config 

def create_pinecone_index():
    load_dotenv() 
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")

    if not PINECONE_API_KEY or not PINECONE_ENV:
        print("PINECONE credentials are not set.")
        return

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = config.PINECONE_INDEX_NAME

    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists. Skipping creation.")
        return

    print(f"Creating new index...")
    try:
        pc.create_index(
            name=index_name,
            dimension=config.EMBEDDING_DIM,
            metric=config.METRIC,
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENV
            )
        )

        max_wait_time = 180  
        start_time = time.time()
        
        while not pc.describe_index(index_name).status['ready']:
            if time.time() - start_time > max_wait_time:
                print("Index creation timeout")
                return
            print("Waiting for index to be ready...")
            time.sleep(5)
            
        print(f"Index '{index_name}' created successfully!")
        
    except Exception as e:
        print(f"Error creating index: {e}")