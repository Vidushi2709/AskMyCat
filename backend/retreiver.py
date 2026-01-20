import os
import argparse
from typing import Any, Dict, List, Tuple
from chromadb import PersistentClient
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
import hashlib
import json
from diskcache import Cache
from datetime import datetime

# Load environment variables from .env
load_dotenv()

# Ranking model (from train.ipynb)
class RankingScorer(nn.Module):
    """Ranking model: outputs embeddings for contrastive ranking."""
    def __init__(self, encoder, embedding_dim=128):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = encoder.config.hidden_size
        
        # Project to embedding space for ranking
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, enc_ips):
        """Return embedding vector for ranking."""
        out = self.encoder(**enc_ips).last_hidden_state
        
        # Mean pooling over tokens
        mask = enc_ips["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, H]
        
        # Project to embedding space
        embedding = self.projection(pooled)  # [B, embedding_dim]
        
        return embedding

class RetrievalPipeline:
    def __init__(self, chroma_path: str = "./collections/ebm", 
                 collection_name: str = "ebm_passages",
                 model_name: str = "distilbert-base-uncased",
                 checkpoint_path: str = "ranking_scorer.ckpt",
                 device: str = "cuda",
                 llm_api_key: str = None,
                 enable_cache: bool = True,
                 cache_dir: str = "./cache"):
            self.device = device
            self.enable_cache = enable_cache
            
            # Initialize cache (retrieval + LLM only)
            if enable_cache:
                self.cache = Cache(cache_dir)
                self.cache_stats = {
                    "retrieval_hits": 0,
                    "retrieval_misses": 0,
                    "llm_hits": 0,
                    "llm_misses": 0,
                }
                print(f"✓ Cache enabled: Retrieval + LLM")
            else:
                self.cache = None
            
            # Chroma client
            self.client = PersistentClient(path=chroma_path)
            try:
                self.collection = self.client.get_collection(collection_name)
                print("✓ Loaded existing collection")
            except Exception as e:
                raise ValueError(f"Collection '{collection_name}' not found in ChromaDB at {chroma_path}. Please ingest data first.") from e
            
            # Initialize tokenizer and encoder
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            encoder = AutoModel.from_pretrained(model_name)
            
            # Load ranking scorer
            self.ranking_model = RankingScorer(encoder, embedding_dim=128).to(device)
            # Use weights_only=True to avoid loading arbitrary pickled objects
            self.ranking_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            self.ranking_model.eval()
            print(f"✓ Loaded ranking model from {checkpoint_path}")
            
            # Initialize LLM (OpenRouter)
            api_key = llm_api_key or os.getenv("API_KEY")
            if not api_key:
                raise ValueError("API_KEY not found in .env file or passed as argument")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"✓ Connected to LLM via OpenRouter ({base_url})")
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        combined = json.dumps((args, sorted(kwargs.items())), sort_keys=True, default=str)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            self.cache_stats = {k: 0 for k in self.cache_stats}
            print("✓ Retrieval + LLM cache cleared")
    
    def print_cache_stats(self):
        """Print cache statistics."""
        if not self.cache:
            print("Cache is disabled")
            return
        
        total_hits = sum(v for k, v in self.cache_stats.items() if "hits" in k)
        total_misses = sum(v for k, v in self.cache_stats.items() if "misses" in k)
        total_requests = total_hits + total_misses
        
        print("\n" + "="*60)
        print("CACHE STATISTICS")
        print("="*60)
        print(f"Retrieval - Hits: {self.cache_stats['retrieval_hits']}, Misses: {self.cache_stats['retrieval_misses']}")
        print(f"LLM       - Hits: {self.cache_stats['llm_hits']}, Misses: {self.cache_stats['llm_misses']}")
        if total_requests > 0:
            hit_rate = (total_hits / total_requests) * 100
            print(f"\nTotal Hit Rate: {hit_rate:.1f}% ({total_hits}/{total_requests})")
        print(f"Cache Size: {len(self.cache)} entries")
        print("="*60 + "\n")
        
    def retrieve_top_k(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.enable_cache:
            cache_key = self._get_cache_key("retrieve", query, top_k)
            if cache_key in self.cache:
                self.cache_stats["retrieval_hits"] += 1
                return self.cache[cache_key]
            self.cache_stats["retrieval_misses"] += 1
        
        results = self.collection.query(query_texts=[query], n_results=top_k)
        passages = results["documents"][0]
        metadatas = results["metadatas"][0]
        result = (passages, metadatas)
        
        if self.enable_cache:
            self.cache[cache_key] = result
            
        print(f"Retrieved {len(passages)} passages from Chroma")
        return result

    def rerank_passages(self, query: str, passages: List[str], metadatas: List[Dict], 
                       max_len: int = 128, batch_size: int = 32) -> List[Tuple[str, Dict, float]]:
        """Rank passages by similarity (no caching - fast operation)."""
        scores = []
        
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type):
            # Encode query
            enc_query = self.tokenizer(
                query,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            enc_query = {k: v.to(self.device) for k, v in enc_query.items()}
            query_embedding = self.ranking_model(enc_query)  # [1, embedding_dim]
            
            # Process passages in batches
            for i in range(0, len(passages), batch_size):
                batch_passages = passages[i:i + batch_size]
                enc_passages = self.tokenizer(
                    batch_passages,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt"
                )
                enc_passages = {k: v.to(self.device) for k, v in enc_passages.items()}
                passage_embeddings = self.ranking_model(enc_passages)  # [batch_size, embedding_dim]
                
                sims = F.cosine_similarity(query_embedding, passage_embeddings)  # [batch_size]
                scores.extend(sims.detach().cpu().tolist())
        
        ranked = sorted(zip(passages, metadatas, scores), key=lambda x: x[2], reverse=True)
        return ranked
    
    def filter_by_threshold(self, ranked_passages: List[Tuple[str, Dict, float]], 
                           threshold: float = 0.5) -> List[Tuple[str, Dict, float]]:
        
        filtered = [p for p in ranked_passages if p[2] >= threshold]
        return filtered
    
    def prepare_for_llm(self, filtered_passages: List[Tuple[str, Dict, float]]) -> str:
        if not filtered_passages:
            return "meow meow, no relevant evidence found."

        seen = set()
        deduped = []
        for passage, meta, score in filtered_passages:
            key = passage.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((passage, meta, score))
            if len(deduped) >= 5:
                break

        context_parts = []
        for i, (passage, metadata, score) in enumerate(deduped, 1):
            context_parts.append(f"[Evidence {i}] (confidence: {score:.3f})\n{passage}")

        return "\n\n".join(context_parts)
    
    def answer_query(self, user_query: str, top_k: int = 10, threshold: float = 0.5, 
                     use_llm: bool = True) -> Dict[str, Any]:
        """Complete end-to-end query pipeline with caching."""
        print(f"\n{'='*80}")
        print(f"Query: {user_query}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve
        passages, metadatas = self.retrieve_top_k(user_query, top_k)
        print(f"[1/4] Retrieved {len(passages)} passages")
        
        # Step 2: Rerank
        ranked = self.rerank_passages(user_query, passages, metadatas)
        print(f"[2/4] Ranked {len(ranked)} passages")
        
        # Step 3: Filter
        filtered = self.filter_by_threshold(ranked, threshold)
        print(f"[3/4] Filtered to {len(filtered)} high-confidence passages")
        
        # Step 4: LLM or context only
        context = self.prepare_for_llm(filtered)
        
        result = {
            "query": user_query,
            "retrieved_count": len(passages),
            "ranked_passages": ranked,
            "filtered_passages": filtered,
            "context": context,
            "answer": None,
            "answer_time": None
        }
        
        if use_llm and filtered:
            print(f"[4/4] Querying LLM...")
            start_time = datetime.now()
            answer = self.query_llm(user_query, context)
            end_time = datetime.now()
            result["answer"] = answer
            result["answer_time"] = (end_time - start_time).total_seconds()
            print(f"[Done] Answer generated in {result['answer_time']:.2f}s")
        else:
            print(f"[4/4] Skipping LLM (use_llm={use_llm}, filtered={len(filtered)>0})")
        
        return result
    
    def query_llm(self, user_query: str, context: str, model: str = "openai/gpt-4o-mini", 
                  temperature: float = 0.7) -> str:
        if self.enable_cache:
            cache_key = self._get_cache_key("llm", user_query, context, model, temperature)
            if cache_key in self.cache:
                self.cache_stats["llm_hits"] += 1
                return self.cache[cache_key]
            self.cache_stats["llm_misses"] += 1
        
        system_prompt = """You are a medical evidence-based medicine (EBM) assistant. 
You answer questions based on provided medical evidence and research.
Always cite the evidence used in your answer.
If evidence is insufficient, clearly state that."""
        
        user_prompt = f"""Based on the following medical evidence, answer this question:

QUESTION: {user_query}

EVIDENCE:
{context}

Please provide a comprehensive, evidence-based answer."""
        
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        
        if self.enable_cache:
            self.cache[cache_key] = answer
        
        return answer
def main():
    """CLI entrypoint: ask a question and talk to the LLM."""
    parser = argparse.ArgumentParser(description="Query the EBM retrieval + LLM pipeline")
    parser.add_argument("query", nargs="?", help="User question to answer")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k passages to retrieve")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold to keep passages")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM call (retrieval + ranking only)")
    args = parser.parse_args()

    # If query not provided as an argument, prompt the user interactively
    if not args.query:
        try:
            args.query = input("Enter your medical question: ").strip()
        except EOFError:
            args.query = None

    if not args.query:
        print("No query provided. Please pass a query argument or type one when prompted.")
        return

    use_llm = not args.no_llm

    pipeline = RetrievalPipeline(
        chroma_path="./collections/ebm",
        collection_name="ebm_passages",
        model_name="distilbert-base-uncased",
        checkpoint_path="ranking_scorer.ckpt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_cache=True,
    )

    result = pipeline.answer_query(
        user_query=args.query,
        top_k=args.top_k,
        threshold=args.threshold,
        use_llm=use_llm,
    )

    print("\n" + "="*80)
    print("ANSWER")
    print("="*80)
    if use_llm and result.get("answer"):
        print(result["answer"])
    else:
        print("LLM skipped; context prepared.")

    print("\n" + "="*80)
    print("EVIDENCE")
    print("="*80)
    print(result.get("context", ""))


if __name__ == "__main__":
    main()