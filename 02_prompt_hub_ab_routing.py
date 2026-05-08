import os
import hashlib
from config import load_config
from qa_pairs import QA_PAIRS
from llm_factory import get_llm, get_embeddings
from vector_store_manager import build_vectorstore

# Load config and set environment variables
config = load_config()
os.environ["LANGCHAIN_TRACING_V2"] = config["langchain_tracing"]
os.environ["LANGCHAIN_API_KEY"] = config["langchain_api_key"] or ""
os.environ["LANGCHAIN_PROJECT"] = config["langchain_project"]

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from prompt_manager import PROMPT_V1, PROMPT_V2

PROJECT_PREFIX = config["langchain_project"]
PROMPT_V1_NAME = f"{PROJECT_PREFIX}-rag-v1"
PROMPT_V2_NAME = f"{PROJECT_PREFIX}-rag-v2"

def push_prompts_to_hub(client):
    print(f"Pushing prompts to Hub...")
    try:
        client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise")
        print(f"✅ Pushed V1")
    except Exception as e: print(f"⚠️ V1: {e}")
    try:
        client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured")
        print(f"✅ Pushed V2")
    except Exception as e: print(f"⚠️ V2: {e}")

def pull_prompts_from_hub(client):
    prompts = {}
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}'")
    except:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️ Fallback for '{PROMPT_V1_NAME}'")
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}'")
    except:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️ Fallback for '{PROMPT_V2_NAME}'")
    return prompts

def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}

def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing (with Fallback)")
    print("=" * 60)

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    for i, qa in enumerate(QA_PAIRS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        try:
            result = ask_ab(retriever, llm, prompts[version_key], qa["question"], version_tag)
            print(f"[{i+1:02d}] [prompt-{version_tag}] {qa['question'][:55]}...")
        except Exception as e:
            print(f"❌ Error on question {i+1}: {e}")

if __name__ == "__main__":
    main()
