import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    
    config = {
        # LangSmith
        "langchain_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
        "langchain_api_key": os.getenv("LANGCHAIN_API_KEY"),
        "langchain_project": os.getenv("LANGCHAIN_PROJECT", "day22-lab-rag"),
        
        # Keys
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "github_token": os.getenv("GITHUB_TOKEN"),
        
        # Endpoints
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "github_base_url": "https://models.inference.ai.azure.com",
        
        # Models
        "openai_model": "gpt-4o-mini",
        "openrouter_model": "openai/gpt-4o-mini",
        "gemini_model": "gemini-1.5-flash",
        "github_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    }
    
    return config

if __name__ == "__main__":
    c = load_config()
    print("✅ Config loaded successfully with GitHub Fallback support")
