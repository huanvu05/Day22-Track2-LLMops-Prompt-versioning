import os
from pathlib import Path
from config import load_config
from qa_pairs import QA_PAIRS
from llm_factory import get_llm, get_embeddings

# Load config and set environment variables
config = load_config()
os.environ["LANGCHAIN_TRACING_V2"] = config["langchain_tracing"]
os.environ["LANGCHAIN_API_KEY"] = config["langchain_api_key"] or ""
os.environ["LANGCHAIN_PROJECT"] = config["langchain_project"]

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from vector_store_manager import build_vectorstore

# ── 3. LLM and Embeddings from Factory ──────────────────────────────────────

# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)

# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline (with Fallback)")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        try:
            answer = ask(chain, question)
            print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {question[:60]}")
            print(f"       A: {answer[:100]}\n")
        except Exception as e:
            print(f"❌ Error on question {i}: {e}")

    print(f"✅ Step 1 complete. Traces in LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")

if __name__ == "__main__":
    main()
