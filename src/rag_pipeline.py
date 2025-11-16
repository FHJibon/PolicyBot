import warnings
warnings.filterwarnings('ignore')

import config
from src.vector_store import get_llm, get_vector_store
from src.system_prompt import system_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

_vector_store = None
_rag_chain = None
_llm = None

def get_initialized_components():
    global _vector_store, _rag_chain, _llm
    
    if _vector_store is None:
        _vector_store = get_vector_store()
    
    if _llm is None:
        _llm = get_llm()
    
    if _rag_chain is None:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question: {question}\n\nContext: {context}")
        ])

        def retrieve_and_format(inputs):
            question = inputs["question"]
            docs = _vector_store.similarity_search(question, k=config.TOP_K)
            context, sources = format_context(docs)
            inputs["_sources"] = sources
            return context

        _rag_chain = (
            RunnablePassthrough.assign(context=retrieve_and_format)
            | prompt_template
            | _llm
            | StrOutputParser()
        )
        print("RAG chain built and cached")
    
    return _rag_chain

def format_context(docs):
    if not docs:
        return "", []
    
    context_parts = []
    most_relevant_source = None
    
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        if content:  
            context_parts.append(content)
        
        if i == 0 and 'source' in doc.metadata:
            source_name = doc.metadata['source'].split('/')[-1].split('\\')[-1]
            page_num = doc.metadata.get('page', 'Unknown')
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            most_relevant_source = f"{source_name} (Page {page_num})"
    
    sources = [most_relevant_source] if most_relevant_source else []
    return "\n\n".join(context_parts), sources

def get_rag_response(query: str, chat_history: list):
    try:
        rag_chain = get_initialized_components()
    
        inputs = {
            "question": query,
            "chat_history": chat_history,
            "_sources": []  
        }
        
        answer = rag_chain.invoke(inputs)
        
        sources = inputs.get("_sources", [])
        
        if not answer or len(answer.strip()) < 10:
            return {
                "answer": "I couldn't find specific information about this in our policy documents. Please consult with HR for detailed guidance on this matter.",
                "sources": []
            }

        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return {
            "answer": "I'm experiencing technical difficulties at the moment. Please try again in a few seconds.",
            "sources": []
        }