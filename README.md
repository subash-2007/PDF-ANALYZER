# DocuMind

DocuMind is an intelligent document Q&A app that lets you upload PDF, DOCX, or TXT files, then ask questions and get highly accurate answers with source context. Powered by OpenAI GPT-4 (or local LLMs), FAISS, and HuggingFace embeddings.

## Features
- Upload PDF, DOCX, or TXT files (single or multiple)
- Extracts and preprocesses file content
- Splits text into chunks and embeds using `all-MiniLM-L6-v2`
- Indexes chunks into a FAISS vector database
- Ask any question about your documents
- Answers with source context and markdown formatting
- Supports OpenAI GPT-4 and local LLMs (Llama2/Mistral)
- Handles large documents (100+ pages)
- Fast response (<2s typical)
- Loading spinner for better UX
- Dynamic file switching

## Tech Stack
- **Language:** Python 3.9+
- **Frontend:** Streamlit
- **LLM:** OpenAI GPT-4 (default), Llama2/Mistral (optional)
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Framework:** LangChain

## Setup
1. **Clone the repo:**
   ```bash
   git clone <your-repo-url>
   cd documind
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your OpenAI API key:**
   - Create a `.env` file or set `OPENAI_API_KEY` in your environment.

4. **Run the app:**
   ```bash
   streamlit run main.py
   ```

## Usage
- Upload one or more documents (PDF, DOCX, TXT)
- Ask any question about the content
- Get answers with source context and markdown formatting
- Switch between uploaded files dynamically

## Configuration
- Edit `config.py` to change model, chunk size, or other settings.

## File Structure
- `main.py`: Streamlit frontend and user flow
- `extractor.py`: Extract and clean text from files
- `qa_engine.py`: Chunking, embedding, vector store, QA pipeline
- `config.py`: Model & path config
- `requirements.txt`: All dependencies listed
- `README.md`: How to run

## License
MIT 