from typing import List, Dict, Any
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from config import config

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def build_faiss_index(chunks: List[str], embedding_model=None) -> FAISS:
    if embedding_model is None:
        embedding_model = get_embedding_model()
    return FAISS.from_texts(chunks, embedding_model)

def save_faiss_index(vectorstore: FAISS, path: str = None):
    if path is None:
        path = config.FAISS_INDEX_PATH
    vectorstore.save_local(path)

def load_faiss_index(path: str = None, embedding_model=None) -> FAISS:
    if path is None:
        path = config.FAISS_INDEX_PATH
    if embedding_model is None:
        embedding_model = get_embedding_model()
    return FAISS.load_local(path, embedding_model)

def get_llm():
    llm_config = config.get_llm_config()
    return LlamaCpp(
        model_path=config.LOCAL_MODEL_PATH,
        **llm_config,
        verbose=True
    )

def get_qa_chain(vectorstore: FAISS, llm=None) -> RetrievalQA:
    if llm is None:
        llm = get_llm()
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question.\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        ),
    )
    
    retriever_config = config.get_retriever_config()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs=retriever_config),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

def answer_question(question: str, vectorstore: FAISS, llm=None) -> Dict[str, Any]:
    qa = get_qa_chain(vectorstore, llm)
    result = qa({"query": question})
    answer = result["result"]
    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
    } for doc in result.get("source_documents", [])]
    return {"answer": answer, "sources": sources}
