import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_core._api.deprecation')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config

def load_and_chunk_pdfs(directory_path):
    documents = []
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return []
    
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s) to process...")
    
    for filename in pdf_files:
        filepath = os.path.join(directory_path, filename)
        print(f"Loading: {filename}")
        
        try:
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num
                        }
                    )
                    documents.append(doc)
            print(f"Processed {filename}: {len(reader.pages)} pages")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
    
    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF(s).")
    
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks