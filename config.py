import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# === LLM Configuration ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # Default to deepseek via OpenRouter
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# === Embedding Model Configuration ===
# Using HuggingFace embeddings (FAISS-compatible)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# === Text Chunking Configuration ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))  # Number of characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))  # Overlap between chunks

# === Vector Store (FAISS) Configuration ===
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

# === Upload Configuration ===
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))
SUPPORTED_FILE_TYPES = os.getenv("SUPPORTED_FILE_TYPES", "pdf,docx,txt").split(",")
