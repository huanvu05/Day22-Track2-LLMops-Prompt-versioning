import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import load_config

config = load_config()

def get_llm(force_gemini=False):
    """
    Returns an LLM with fallback logic:
    OpenAI -> GitHub (10 req limit) -> OpenRouter -> Gemini
    """
    
    # 1. Primary: OpenAI
    openai_llm = ChatOpenAI(
        model=config["openai_model"],
        api_key=config["openai_api_key"],
        base_url=config["openai_base_url"],
        max_retries=0
    )
    
    # 2. Secondary: GitHub Models (Very low limit: 10 requests)
    github_llm = ChatOpenAI(
        model=config["github_model"],
        api_key=config["github_token"],
        base_url=config["github_base_url"],
        max_retries=0
    )
    
    # 3. Tertiary: OpenRouter
    openrouter_llm = ChatOpenAI(
        model=config["openrouter_model"],
        api_key=config["openrouter_api_key"],
        base_url=config["openrouter_base_url"],
        max_retries=0
    )
    
    # 4. Final Fallback: Gemini (Free & High limit)
    gemini_llm = ChatGoogleGenerativeAI(
        model=config["gemini_model"],
        google_api_key=config["gemini_api_key"],
    )
    
    if force_gemini:
        return gemini_llm

    # Fallback chain: OpenAI -> GitHub -> OpenRouter -> Gemini
    # Note: GitHub has 10 req limit, so it will fail fast and move to OpenRouter/Gemini
    return openai_llm.with_fallbacks([github_llm, openrouter_llm, gemini_llm])

def get_embeddings():
    """
    Returns OpenRouter Embeddings. 
    If OpenRouter credits are 0, falls back to OpenAI base (which might fail with 429).
    """
    if config["openrouter_api_key"]:
        return OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            api_key=config["openrouter_api_key"],
            base_url="https://openrouter.ai/api/v1"
        )
    
    return OpenAIEmbeddings(
        model=config["embedding_model"],
        api_key=config["openai_api_key"],
        base_url=config["openai_base_url"]
    )
