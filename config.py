import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# App Configuration Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
APP_URL = os.getenv("APP_URL", "http://localhost:3000")

# Vector Database Path
CHROMA_STORE_DIR = os.getenv("CHROMA_STORE_DIR", "chroma_store")

def validate_config():
    """
    Validates that necessary configuration properties are present.
    Returns a tuple of (is_valid, list_of_warnings).
    """
    warnings = []
    if not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY is not set. Gemini API functions will fail.")
    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY is not set. External web search features will fail.")
    
    return len(warnings) == 0, warnings
