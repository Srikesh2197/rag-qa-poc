import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "llama3.1:8b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 3
