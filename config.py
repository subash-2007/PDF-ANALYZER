import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for DocuMind application."""
    
    def __init__(self):
        # LLM Configuration
        self.LLM_PROVIDER = "local"
        
        # Embedding Configuration
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Chunking Configuration
        self.CHUNK_SIZE = 500
        self.CHUNK_OVERLAP = 50
        
        # FAISS Configuration
        self.FAISS_INDEX_PATH = "./faiss_index"
        
        # File Upload Configuration
        self.MAX_UPLOAD_SIZE_MB = 50
        self.SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]
        
        # Local LLM Configuration
        self.LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./models/tinyllama.gguf")
        
        # API Keys (optional for local deployment)
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
        
        # LLM Parameters
        self.TEMPERATURE = 0
        self.MAX_TOKENS = 512
        self.N_CTX = 2048
        self.N_BATCH = 8
        
        # Retrieval Configuration
        self.RETRIEVER_K = 4
        
    def get_llm_config(self):
        """Get LLM configuration dictionary."""
        return {
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
            "n_ctx": self.N_CTX,
            "n_batch": self.N_BATCH
        }
    
    def get_retriever_config(self):
        """Get retriever configuration dictionary."""
        return {
            "k": self.RETRIEVER_K
        }
    
    def validate_config(self):
        """Validate configuration settings."""
        if not os.path.exists(self.LOCAL_MODEL_PATH):
            print(f"Warning: Local model not found at {self.LOCAL_MODEL_PATH}")
        
        if not os.path.exists(self.FAISS_INDEX_PATH):
            os.makedirs(self.FAISS_INDEX_PATH, exist_ok=True)
            print(f"Created FAISS index directory: {self.FAISS_INDEX_PATH}")

# Create a global config instance
config = Config()

# For backward compatibility, export all config attributes
LLM_PROVIDER = config.LLM_PROVIDER
EMBEDDING_MODEL = config.EMBEDDING_MODEL
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
MAX_UPLOAD_SIZE_MB = config.MAX_UPLOAD_SIZE_MB
SUPPORTED_FILE_TYPES = config.SUPPORTED_FILE_TYPES
LOCAL_MODEL_PATH = config.LOCAL_MODEL_PATH
GEMINI_API_KEY = config.GEMINI_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
TEMPERATURE = config.TEMPERATURE
MAX_TOKENS = config.MAX_TOKENS
N_CTX = config.N_CTX
N_BATCH = config.N_BATCH
RETRIEVER_K = config.RETRIEVER_K
