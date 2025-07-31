from typing import List, Dict, Any
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH,
    DEEPSEEK_API_KEY
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embedding_model():
    """Always use HuggingFace embedding model (FAISS-compatible)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_faiss_index(chunks: List[str], embedding_model=None) -> FAISS:
    """Embed chunks and build FAISS index."""
    if embedding_model is None:
        embedding_model = get_embedding_model()
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    return vectorstore


def save_faiss_index(vectorstore: FAISS, path: str = FAISS_INDEX_PATH):
    """Save FAISS index to disk."""
    vectorstore.save_local(path)


def load_faiss_index(path: str = FAISS_INDEX_PATH, embedding_model=None) -> FAISS:
    """Load FAISS index from disk."""
    if embedding_model is None:
        embedding_model = get_embedding_model()
    return FAISS.load_local(path, embedding_model)


def get_llm() -> BaseChatModel:
    """Return DeepSeek model via LangChain."""
    from langchain_community.chat_models.deepseek import ChatDeepSeek
    return ChatDeepSeek(
        model="deepseek-chat",  # or deepseek-coder
        api_key=DEEPSEEK_API_KEY,
        temperature=0
    )



def get_qa_chain(vectorstore: FAISS, llm: BaseChatModel = None) -> RetrievalQA:
    """Create a QA chain with retrieval and source context."""
    if llm is None:
        llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use the following context to answer the question.\n"
            "If the answer is not in the context, say 'I don't know.'\n"
            "Return the answer in markdown format.\n"
            "\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        ),
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa


def answer_question(
    question: str, vectorstore: FAISS, llm: BaseChatModel = None
) -> Dict[str, Any]:
    """Answer a question using the QA chain and return answer with source context."""
    qa = get_qa_chain(vectorstore, llm)
    result = qa({"query": question})
    answer = result["result"]
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
        })
    return {"answer": answer, "sources": sources}
