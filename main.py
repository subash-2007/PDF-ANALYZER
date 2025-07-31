import os
import tempfile
import streamlit as st
from extractor import extract_text, clean_text, SUPPORTED_FILE_TYPES
from qa_engine import (
    chunk_text, build_faiss_index, save_faiss_index, load_faiss_index, answer_question, get_llm
)
from config import MAX_UPLOAD_SIZE_MB

st.set_page_config(page_title="DocuMind", layout="wide")
st.title("🧠 DocuMind: Document Q&A")

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True,
    help=f"Max {MAX_UPLOAD_SIZE_MB}MB per file."
)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    selected_file = st.sidebar.selectbox("Select file to query", file_names)
    file_map = {f.name: f for f in uploaded_files}
    st.sidebar.write(f"**{len(uploaded_files)} file(s) uploaded.**")
else:
    st.info("Upload one or more documents to get started.")
    st.stop()

# Session state for vectorstores
if 'vectorstores' not in st.session_state:
    st.session_state['vectorstores'] = {}
if 'texts' not in st.session_state:
    st.session_state['texts'] = {}

# Process selected file
file_obj = file_map[selected_file]
if selected_file not in st.session_state['vectorstores']:
    with st.spinner(f"Extracting and indexing '{selected_file}'..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(selected_file)[1]) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        raw_text = extract_text(tmp_path)
        cleaned_text = clean_text(raw_text)
        st.session_state['texts'][selected_file] = cleaned_text
        chunks = chunk_text(cleaned_text)
        vectorstore = build_faiss_index(chunks)
        st.session_state['vectorstores'][selected_file] = vectorstore
        os.unlink(tmp_path)
else:
    vectorstore = st.session_state['vectorstores'][selected_file]
    cleaned_text = st.session_state['texts'][selected_file]

st.success(f"Ready to answer questions about '{selected_file}'!")

# Question input
st.subheader("Ask a question about your document:")
def_qa = "What is this document about?"
question = st.text_input("Your question", value=def_qa, key="question_input")

if st.button("Get Answer", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        llm = get_llm()
        result = answer_question(question, vectorstore, llm)
        st.markdown(f"### Answer\n{result['answer']}")
        if result['sources']:
            st.markdown("---")
            st.markdown("#### Source Context:")
            for i, src in enumerate(result['sources']):
                st.markdown(f"""**Chunk {i+1}:**\n```
{src['content'][:500]}
```""")
else:
    st.info("Enter a question and click 'Get Answer'.")

st.sidebar.markdown("---")
st.sidebar.markdown("[View on GitHub](#)") 