import sys
from llm_factory import get_llm, get_embeddings

def test():
    print("🔍 Testing API Setup...")
    try:
        print("1. Testing Embeddings...")
        emb = get_embeddings()
        vector = emb.embed_query("Hello world")
        print(f"   ✅ Embeddings OK (Dim: {len(vector)})")
        
        print("2. Testing LLM Fallback...")
        llm = get_llm()
        res = llm.invoke("Say 'Hello API'")
        print(f"   ✅ LLM OK: {res.content}")
        
        print("\n🚀 All systems ready! You can now run 'python run_all.py'")
    except Exception as e:
        print(f"\n❌ Setup Test Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test()
