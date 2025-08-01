import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import qa_engine
import extractor
from config import config

# Page configuration
st.set_page_config(
    page_title="DocuMind - Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

def main():
    # Header
    st.title("📚 DocuMind - Intelligent Document Q&A")
    st.markdown("Upload your documents and ask questions to get intelligent answers with source context.")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=config.SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(config.SUPPORTED_FILE_TYPES).upper()}"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            # File selection
            if len(uploaded_files) > 1:
                file_names = [f.name for f in uploaded_files]
                selected_file = st.selectbox("Select file to process:", file_names)
                st.session_state.current_file = selected_file
            else:
                st.session_state.current_file = uploaded_files[0].name
            
            # Process button
            if st.button("🔄 Process Documents", type="primary"):
                process_documents(uploaded_files)
    
    # Main content area
    if st.session_state.vectorstore is None:
        st.info("👆 Please upload and process documents to start asking questions.")
        show_features()
    else:
        show_qa_interface()

def process_documents(uploaded_files):
    """Process uploaded documents and create vector store."""
    with st.spinner("Processing documents..."):
        try:
            # Extract text from all files
            all_text = ""
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Extract text
                    text = extractor.extract_text(tmp_file_path)
                    text = extractor.clean_text(text)
                    all_text += f"\n\n--- {uploaded_file.name} ---\n\n{text}"
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
            
            # Chunk text
            chunks = qa_engine.chunk_text(all_text)
            
            # Build vector store
            vectorstore = qa_engine.build_faiss_index(chunks)
            
            # Save vector store
            qa_engine.save_faiss_index(vectorstore)
            
            # Store in session state
            st.session_state.vectorstore = vectorstore
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} document(s) with {len(chunks)} chunks!")
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")

def show_qa_interface():
    """Show the Q&A interface."""
    st.header("🤖 Ask Questions")
    
    # Question input
    question = st.text_input(
        "What would you like to know about your documents?",
        placeholder="e.g., What are the main topics discussed?",
        key="question_input"
    )
    
    if question and st.button("🔍 Get Answer", type="primary"):
        with st.spinner("Generating answer..."):
            try:
                # Get answer
                result = qa_engine.answer_question(question, st.session_state.vectorstore)
                
                # Display answer
                st.subheader("💡 Answer")
                st.markdown(result["answer"])
                
                # Display sources
                if result["sources"]:
                    st.subheader("📖 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"Source {i}"):
                            st.markdown(source["content"])
                            if source["metadata"]:
                                st.caption(f"Metadata: {source['metadata']}")
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")

def show_features():
    """Show application features."""
    st.markdown("---")
    st.subheader("✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📄 Multi-Format Support**
        - PDF documents
        - Word documents (DOCX)
        - Text files (TXT)
        """)
    
    with col2:
        st.markdown("""
        **🧠 Intelligent Processing**
        - Advanced text chunking
        - Semantic embeddings
        - FAISS vector search
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI-Powered Q&A**
        - Local LLM support
        - Context-aware answers
        - Source citations
        """)

def validate_environment():
    """Validate the environment and configuration."""
    config.validate_config()
    
    # Check if local model exists
    if not os.path.exists(config.LOCAL_MODEL_PATH):
        st.warning(f"⚠️ Local model not found at {config.LOCAL_MODEL_PATH}")
        st.info("Please download a GGUF model and update the LOCAL_MODEL_PATH in config.py")

if __name__ == "__main__":
    validate_environment()
    main()
