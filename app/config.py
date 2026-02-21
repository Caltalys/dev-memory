import os
from dotenv import load_dotenv
from pathlib import Path
import logging

load_dotenv()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DevMemory")


class Settings:
    NOTES_DIR = Path(os.getenv("NOTES_DIR", "./data/notes"))
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")
    DB_PATH = os.getenv("DB_PATH", "./data/dev_memory.db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K = int(os.getenv("TOP_K", 5))
    NUM_CTX = int(os.getenv("NUM_CTX", 4096))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))


settings = Settings()

# Tạo thư mục nếu chưa tồn tại
settings.NOTES_DIR.mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
