import json
import time
import numpy as np
from sklearn.metrics import ndcg_score
from backend import NustCopilot
import config

# --- CONFIGURATION ---
RETRIEVAL_SAMPLE_SIZE = 120  # Run ALL 120 for Recall (Free/Local)
GENERATION_SAMPLE_SIZE = 5   # Run only 5 for LLM Generation (To save Quota)
# ---------------------

# Load Data
try:
    with open("eval_set.json", "r", encoding="utf-8") as f:
        EVAL_DATA = json.load(f)
except FileNotFoundError:
    print("Error: eval_set.json not found.")
    exit()

bot = NustCopilot()

def calculate_retrieval_metrics(retrieved_docs, gold_ids):
    """Calculates Recall and nDCG (No LLM required)"""
    retrieved_ids = [d.metadata['doc_id'] for d in retrieved_docs]
    hits = set(retrieved_ids).intersection(set(gold_ids))
    recall = 1.0 if hits else 0.0
    
    relevance = [1 if rid in gold_ids else 0 for rid in retrieved_ids]
    if len(relevance) < config.RETRIEVAL_K:
        relevance += [0] * (config.RETRIEVAL_K - len(relevance))
    
    ideal = sorted(relevance, reverse=True)
    try:
        ndcg = ndcg_score([ideal], [relevance], k=config.RETRIEVAL_K)
    except:
        ndcg = 0.0
    return recall, ndcg

def calculate_faithfulness(generated_answer, retrieved_docs):
    """Calculates Citation Precision (Requires LLM output)"""
    import re
    retrieved_ids = [d.metadata['doc_id'] for d in retrieved_docs]
    citations = re.findall(r'\[(.*?)\]', generated_answer)
    if not citations: return 1.0 # Faithful by abstention
    
    valid = sum(1 for c in citations if c in retrieved_ids)
    return valid / len(citations)

def count_unsupported_citations(generated_answer, retrieved_docs):
    """Number of citations that strict mode would strip (cited ID not among retrieved docs)"""
    import re
    retrieved_ids = [d.metadata['doc_id'] for d in retrieved_docs]
    citations = re.findall(r'\[(.*?)\]', generated_answer)
    return sum(1 for c in citations if c not in retrieved_ids)

def run_smart_ablation(mode_name, use_hybrid):
    print(f"\n--- Running Study: {mode_name} ---")
    results = {"recall": [], "ndcg": [], "citation_precision": [], "unsupported": [], "latency": []}
    
    # 1. RETRIEVAL PHASE (Runs on ALL 120 questions - It is FREE/LOCAL)
    print(f"   -> Phase 1: Retrieval Eval on {RETRIEVAL_SAMPLE_SIZE} questions (Local/Fast)...")
    for i in range(RETRIEVAL_SAMPLE_SIZE):
        if i >= len(EVAL_DATA): break
        
        item = EVAL_DATA[i]
        query = item['question']
        gold_ids = item['gold_ids']
        
        # Retrieval only (No API Call)
        start = time.time()
        if use_hybrid:
            docs = bot.get_hybrid_docs(query)
        else:
            docs = bot.dense_retriever.invoke(query)
        lat = time.time() - start
        
        rec, ndcg = calculate_retrieval_metrics(docs, gold_ids)
        results["recall"].append(rec)
        results["ndcg"].append(ndcg)
        
    # 2. GENERATION PHASE (Runs on SAMPLE only - Uses API)
    # Faithfulness is scored on the RAW LLM output. Scoring after strict mode would
    # always give 1.0, since strict mode strips exactly the citations this metric counts.
    print(f"   -> Phase 2: Generation Eval on {GENERATION_SAMPLE_SIZE} questions (API usage)...")
    for i in range(GENERATION_SAMPLE_SIZE):
        item = EVAL_DATA[i]
        query = item['question']
        
        try:
            # Add sleep to prevent rate limit (429) even on small batches
            time.sleep(5) 
            start = time.time()
            response, docs = bot.generate(query, use_hybrid=use_hybrid)
            lat = time.time() - start
            
            results["citation_precision"].append(calculate_faithfulness(response, docs))
            results["unsupported"].append(count_unsupported_citations(response, docs))
            results["latency"].append(lat)
            print(f"      [Generation {i+1}/{GENERATION_SAMPLE_SIZE}] Success")
            
        except Exception as e:
            print(f"      [Generation {i+1}] API Error: {e}")
            time.sleep(10) # Wait longer if error

    # Aggregates
    print(f"\nRESULTS FOR {mode_name}:")
    print(f"Avg Recall@5 (N={len(results['recall'])}): {np.mean(results['recall']):.4f}")
    print(f"Avg nDCG@5 (N={len(results['ndcg'])}):   {np.mean(results['ndcg']):.4f}")
    if results["citation_precision"]:
        n = len(results["citation_precision"])
        flagged = sum(1 for u in results["unsupported"] if u > 0)
        print(f"Avg Cit Prec, raw LLM output (N={n}): {np.mean(results['citation_precision']):.4f}")
        print(f"Answers with unsupported citations (strict mode would strip): {flagged}/{n} "
              f"({sum(results['unsupported'])} citations total)")
        print(f"Avg Latency  (N={len(results['latency'])}):  {np.mean(results['latency']):.4f}s")
    else:
        print("Avg Cit Prec: N/A (API Limit)")

def run_staleness_check():
    print("\n--- Running Study: Staleness & Abstention ---")
    tricks = [
        "What was the fee structure in 2018?",
        "What is the policy for space travel?",
    ]
    for t in tricks:
        try:
            ans, _ = bot.answer(t, strict_mode=True)
            print(f"Q: {t}")
            print(f"A: {ans}\n")
            time.sleep(5)
        except Exception as e:
            print(f"Error on trick question: {e}")

if __name__ == "__main__":
    # --- STUDY 1: Retrieval Impact (Dense vs Hybrid) ---
    # Each run also reports faithfulness on the raw LLM output, i.e. what users would see
    # with Strict Mode OFF, plus how many citations Strict Mode catches (STUDY 2).
    run_smart_ablation("Dense Retrieval Only", use_hybrid=False)
    run_smart_ablation("Hybrid Retrieval (Final)", use_hybrid=True)

    # --- STUDY 3: Staleness/Abstention ---
    run_staleness_check()