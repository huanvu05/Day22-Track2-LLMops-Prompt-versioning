import os
import json
import warnings
import numpy as np
import uuid
import builtins
from pathlib import Path

# Fix for RAGAS internal parallel jobs
builtins.uuid = uuid

from config import load_config
from qa_pairs import QA_PAIRS
from llm_factory import get_llm, get_embeddings
from vector_store_manager import build_vectorstore
from prompt_manager import PROMPT_V1, PROMPT_V2

warnings.filterwarnings("ignore")
config = load_config()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_core.output_parsers import StrOutputParser
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = (prompt | llm | StrOutputParser()).invoke({"context": "\n\n".join(contexts), "question": question})
    return {"answer": answer, "contexts": contexts}

def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm() # Standard fallback for generation
    prompt = PROMPTS[prompt_version]
    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
            results.append({
                "question":  qa["question"],
                "reference": qa["reference"],
                "answer":    out["answer"],
                "contexts":  out["contexts"],
            })
            if i % 10 == 0: print(f"  [{i:02d}/50] processed...")
        except Exception as e:
            print(f"  ❌ Error on {i}: {e}")
    return results

def build_ragas_dataset(rag_results: list):
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)

def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    
    # Use Gemini directly for evaluation to save OpenRouter credits
    llm_base = get_llm(force_gemini=True)
    emb_base = get_embeddings()
    
    llm_eval = LangchainLLMWrapper(langchain_llm=llm_base)
    emb_eval = LangchainEmbeddingsWrapper(embeddings=emb_base)
    
    run_config = RunConfig(max_workers=1)
    
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm_eval,
            embeddings=emb_eval,
            run_config=run_config
        )
        
        scores = {}
        for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            raw = result[key]
            # Use np.nanmean to ignore failed samples (nan)
            scores[key] = float(np.nanmean(raw))
            print(f"  {key:30s}: {scores[key]:.4f}")
            
        return scores
    except Exception as e:
        print(f"❌ RAGAS Evaluation failed: {e}")
        return {k: 0.0 for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]}

def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation (Stable Mode)")
    print("=" * 60)
    vectorstore = build_vectorstore()
    
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")
    
    print("\n" + "-" * 60)
    print(f"{'Metric':30s} | {'V1':8s} | {'V2':8s} | {'Winner'}")
    print("-" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2"
        print(f"{metric:30s} | {s1:.4f} | {s2:.4f} | {winner}")

    report = {"prompt_v1_scores": v1_scores, "prompt_v2_scores": v2_scores}
    Path("data/ragas_report.json").write_text(json.dumps(report, indent=2))
    print("\n💾 Saved data/ragas_report.json")

if __name__ == "__main__":
    main()
