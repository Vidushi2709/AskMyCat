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
import sys
from pathlib import Path
import threading
import logging

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PATH, CHROMA_COLLECTION_NAME, MODEL_NAME,
    RANKING_SCORER_CKPT, CACHE_DIR
)

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    def __init__(self, chroma_path: str = None, 
                 collection_name: str = None,
                 model_name: str = None,
                 checkpoint_path: str = None,
                 device: str = "cuda",
                 llm_api_key: str = None,
                 enable_cache: bool = True,
                 cache_dir: str = None):
            # Use config defaults if not provided
            chroma_path = chroma_path or str(CHROMA_PATH)
            collection_name = collection_name or CHROMA_COLLECTION_NAME
            model_name = model_name or MODEL_NAME
            checkpoint_path = checkpoint_path or str(RANKING_SCORER_CKPT)
            cache_dir = cache_dir or str(CACHE_DIR)
            
            self.device = device
            self.enable_cache = enable_cache
            
            # Initialize cache (retrieval + LLM only)
            if enable_cache:
                # Cache with size limit (1GB) and LRU eviction policy
                self.cache = Cache(cache_dir, 
                                   size_limit=int(1e9),  # 1GB max
                                   eviction_policy='least-recently-used')
                # Set TTL for cache entries (7 days in seconds)
                self.cache_ttl = 7 * 24 * 60 * 60
                self.cache_stats = {
                    "retrieval_hits": 0,
                    "retrieval_misses": 0,
                    "llm_hits": 0,
                    "llm_misses": 0,
                }
                logger.info("Cache enabled: Retrieval + LLM (1GB max, 7-day TTL)")
            else:
                self.cache = None
                self.cache_ttl = None
            
            # Chroma client
            self.client = PersistentClient(path=chroma_path)
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info("Loaded existing collection")
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
            logger.info(f"Loaded ranking model from {checkpoint_path}")
            
            # Initialize LLM (OpenRouter)
            api_key = llm_api_key or os.getenv("API_KEY")
            if not api_key:
                raise ValueError("API_KEY not found in .env file or passed as argument")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Connected to LLM via OpenRouter ({base_url})")
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        combined = json.dumps((args, sorted(kwargs.items())), sort_keys=True, default=str)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            self.cache_stats = {k: 0 for k in self.cache_stats}
            logger.info("Retrieval + LLM cache cleared")
    
    def print_cache_stats(self):
        """Print cache statistics."""
        if not self.cache:
            logger.info("Cache is disabled")
            return
        
        total_hits = sum(v for k, v in self.cache_stats.items() if "hits" in k)
        total_misses = sum(v for k, v in self.cache_stats.items() if "misses" in k)
        total_requests = total_hits + total_misses
        
        logger.info("="*60)
        logger.info("CACHE STATISTICS")
        logger.info("="*60)
        logger.info(f"Retrieval - Hits: {self.cache_stats['retrieval_hits']}, Misses: {self.cache_stats['retrieval_misses']}")
        logger.info(f"LLM       - Hits: {self.cache_stats['llm_hits']}, Misses: {self.cache_stats['llm_misses']}")
        if total_requests > 0:
            hit_rate = (total_hits / total_requests) * 100
            logger.info(f"Total Hit Rate: {hit_rate:.1f}% ({total_hits}/{total_requests})")
        logger.info(f"Cache Size: {len(self.cache)} entries")
        logger.info("="*60)
        
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
        
        # Cache with TTL
        if self.enable_cache:
            self.cache.set(cache_key, result, expire=self.cache_ttl)
            
        logger.debug(f"Retrieved {len(passages)} passages from Chroma")
        return result
    
    def rerank_passages(self, query: str, passages: List[str], metadatas: List[Dict[str, Any]],
                       max_len: int = 128, batch_size: int = 32, timeout: int = 30) -> List[Tuple[str, Dict, float]]:
        """Rerank passages using the trained ranking model with timeout protection."""
        result_container = {"ranked": None, "error": None}
        
        def rerank_worker():
            """Worker thread that performs reranking."""
            try:
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
                
                result_container["ranked"] = sorted(zip(passages, metadatas, scores), key=lambda x: x[2], reverse=True)
            except Exception as e:
                result_container["error"] = e
        
        # Run reranking in a thread with timeout
        worker_thread = threading.Thread(target=rerank_worker, daemon=True)
        worker_thread.start()
        worker_thread.join(timeout=timeout)
        
        # Check if thread is still alive (timeout occurred)
        if worker_thread.is_alive():
            raise TimeoutError(f"Reranking took longer than {timeout} seconds for {len(passages)} passages")
        
        # Check for errors that occurred in the worker thread
        if result_container["error"] is not None:
            raise result_container["error"]
        
        return result_container["ranked"]
    
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
    
    # gate 1: query quality check
    def gate_1_query_quality(self, user_query: str, threshold: float = 0.7) -> Dict[str, Any]:
        logger.info("Gate 1: Checking query quality...")
        
        # Simple heuristics for query quality
        energy_score = 1.0
        reasons = []
        
        # Check query length
        if len(user_query.strip()) < 10:
            energy_score -= 0.3
            reasons.append("Query is too short")
        
        # Check if it's a question or statement
        question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who', 'can', 'should', 'is', 'are', 'does']
        has_question = any(word in user_query.lower().split()[:5] for word in question_words) or '?' in user_query
        if not has_question:
            energy_score -= 0.2
            reasons.append("Query doesn't appear to be a clear question")
        
        # Check for overly broad queries
        broad_words = ['everything', 'anything', 'all about', 'tell me about', 'explain']
        if any(word in user_query.lower() for word in broad_words):
            energy_score -= 0.1
            reasons.append("Query may be too broad")
        
        # Check for multiple questions
        if user_query.count('?') > 1:
            energy_score -= 0.15
            reasons.append("Multiple questions detected - consider asking separately")
        
        passed = energy_score >= threshold
        
        result = {
            "gate": "query_quality",
            "passed": passed,
            "energy_score": max(0.0, energy_score),
            "threshold": threshold,
            "reasons": reasons if not passed else ["Query appears clear and specific"],
            "action": "proceed" if passed else "reject"
        }
        
        if passed:
            logger.info(f"Gate 1 PASSED: Query quality = {result['energy_score']:.2f}")
        else:
            logger.warning(f"Gate 1 FAILED: Query quality = {result['energy_score']:.2f} (threshold: {threshold})")
        
        return result
    
    # gate 2: retrieval quality check 
    def gate_2_retrieval_quality(self, passages: List[str], metadatas: List[Dict], 
                                 threshold: float = 0.6) -> Dict[str, Any]:
        logger.info("Gate 2: Checking retrieval quality...")
        
        energy_score = 1.0
        reasons = []
        
        # Check if we got any results
        if not passages or len(passages) == 0:
            return {
                "gate": "retrieval_quality",
                "passed": False,
                "energy_score": 0.0,
                "threshold": threshold,
                "reasons": ["No evidence found in database"],
                "action": "reject"
            }
        
        # Check number of results
        if len(passages) < 3:
            energy_score -= 0.3
            reasons.append(f"Only {len(passages)} passages found - limited evidence")
        elif len(passages) < 5:
            energy_score -= 0.1
            reasons.append(f"Moderate evidence: {len(passages)} passages")
        else:
            reasons.append(f"Good coverage: {len(passages)} passages found")
        
        # Check passage length (too short = low quality)
        avg_length = sum(len(p) for p in passages) / len(passages)
        if avg_length < 100:
            energy_score -= 0.2
            reasons.append("Passages are very short - may lack context")
        elif avg_length < 200:
            energy_score -= 0.1
            reasons.append("Passages are somewhat short")
        
        # Check for duplicate or near-duplicate passages
        unique_passages = set(p.strip().lower()[:200] for p in passages)
        duplicate_ratio = 1 - (len(unique_passages) / len(passages))
        if duplicate_ratio > 0.3:
            energy_score -= 0.15
            reasons.append(f"High duplication rate ({duplicate_ratio:.0%})")
        
        passed = energy_score >= threshold
        
        result = {
            "gate": "retrieval_quality",
            "passed": passed,
            "energy_score": max(0.0, energy_score),
            "threshold": threshold,
            "reasons": reasons,
            "action": "proceed" if passed else "reject"
        }
        
        if passed:
            logger.info(f" Gate 2 PASSED: Retrieval quality = {result['energy_score']:.2f}")
        else:
            logger.warning(f" Gate 2 FAILED: Retrieval quality = {result['energy_score']:.2f} (threshold: {threshold})")
        
        return result
    
    # gate 3: evidence consistency check
    def gate_3_evidence_consistency(self, filtered_passages: List[Tuple[str, Dict, float]], 
                                   threshold: float = 0.65) -> Dict[str, Any]:
        logger.info("Gate 3: Checking evidence consistency...")
        
        energy_score = 1.0
        reasons = []
        
        # Check if we have enough high-quality evidence
        if not filtered_passages or len(filtered_passages) == 0:
            return {
                "gate": "evidence_consistency",
                "passed": False,
                "energy_score": 0.0,
                "threshold": threshold,
                "reasons": ["No passages passed confidence threshold"],
                "action": "reject"
            }
        
        # Check number of filtered passages
        if len(filtered_passages) < 2:
            energy_score -= 0.4
            reasons.append(f"Only {len(filtered_passages)} passage with sufficient confidence - CRITICAL")
        elif len(filtered_passages) < 3:
            energy_score -= 0.2
            reasons.append(f"Limited evidence: {len(filtered_passages)} passages")
        else:
            reasons.append(f"Multiple sources: {len(filtered_passages)} passages")
        
        # Check average similarity score
        avg_similarity = sum(score for _, _, score in filtered_passages) / len(filtered_passages)
        if avg_similarity < 0.5:
            energy_score -= 0.3
            reasons.append(f"Low average similarity ({avg_similarity:.2f})")
        elif avg_similarity < 0.7:
            energy_score -= 0.1
            reasons.append(f"Moderate average similarity ({avg_similarity:.2f})")
        else:
            reasons.append(f"Strong similarity ({avg_similarity:.2f})")
        
        # Check score variance (high variance = inconsistent evidence)
        if len(filtered_passages) > 1:
            scores = [score for _, _, score in filtered_passages]
            score_variance = sum((s - avg_similarity) ** 2 for s in scores) / len(scores)
            if score_variance > 0.1:
                energy_score -= 0.15
                reasons.append("High score variance - evidence may be inconsistent")
        
        # Check if top passage is significantly better than others
        if len(filtered_passages) > 1:
            top_score = filtered_passages[0][2]
            second_score = filtered_passages[1][2]
            if top_score - second_score > 0.3:
                energy_score -= 0.1
                reasons.append("Large gap between top passages - confidence concentrated in one source")
        
        passed = energy_score >= threshold
        
        result = {
            "gate": "evidence_consistency",
            "passed": passed,
            "energy_score": max(0.0, energy_score),
            "threshold": threshold,
            "reasons": reasons,
            "action": "proceed" if passed else "reject",
            "metrics": {
                "num_passages": len(filtered_passages),
                "avg_similarity": round(avg_similarity, 3)
            }
        }
        
        if passed:
            logger.info(f" Gate 3 PASSED: Evidence consistency = {result['energy_score']:.2f}")
        else:
            logger.warning(f" Gate 3 FAILED: Evidence consistency = {result['energy_score']:.2f} (threshold: {threshold})")
        
        return result
    
    # verify each evidence with answer 
    def verify_evidence_chain(self, answer: str, filtered_passages: List[Tuple[str, Dict, float]],
                             min_similarity: float = 0.6) -> Dict[str, Any]:
        logger.info(" Verifying evidence chain...")
        
        if not answer or not filtered_passages:
            return {
                "verified": False,
                "total_sentences": 0,
                "verified_sentences": 0,
                "unverified_sentences": 0,
                "verification_rate": 0.0,
                "sentences": [],
                "warning": "No answer or evidence to verify"
            }
        
        # Split answer into sentences (simple split on .!?)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return {
                "verified": False,
                "total_sentences": 0,
                "verified_sentences": 0,
                "unverified_sentences": 0,
                "verification_rate": 0.0,
                "sentences": [],
                "warning": "Could not parse answer into sentences"
            }
        
        logger.info(f"Analyzing {len(sentences)} sentences against {len(filtered_passages)} evidence passages")
        
        # Helper function to encode text
        def encode_text(text: str, max_len: int = 128):
            """Encode text using the ranking model."""
            with torch.no_grad():
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                embedding = self.ranking_model(enc)  # [1, embedding_dim]
                return embedding.cpu().numpy()[0]  # Return as numpy array
        
        # Encode all passages once
        passage_texts = [passage for passage, _, _ in filtered_passages]
        passage_embeddings = [encode_text(text) for text in passage_texts]
        
        verification_results = []
        verified_count = 0
        
        for sentence_idx, sentence in enumerate(sentences, 1):
            # Encode sentence
            sentence_emb = encode_text(sentence)
            
            # Calculate similarity with all passages
            similarities = []
            for i, passage_emb in enumerate(passage_embeddings):
                sim_score = F.cosine_similarity(
                    torch.tensor(sentence_emb).unsqueeze(0),
                    torch.tensor(passage_emb).unsqueeze(0)
                ).item()
                similarities.append((i, sim_score))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Find best matching passage(s)
            supporting_passages = []
            for passage_idx, sim_score in similarities[:3]:  # Top 3 matches
                if sim_score >= min_similarity:
                    supporting_passages.append({
                        "passage_number": passage_idx + 1,
                        "similarity": round(sim_score, 3),
                        "passage_preview": passage_texts[passage_idx][:150] + "...",
                        "metadata": filtered_passages[passage_idx][1]
                    })
            
            # Determine if verified
            is_verified = len(supporting_passages) > 0
            if is_verified:
                verified_count += 1
            
            verification_results.append({
                "sentence_number": sentence_idx,
                "sentence": sentence,
                "verified": is_verified,
                "confidence": supporting_passages[0]["similarity"] if supporting_passages else 0.0,
                "supporting_passages": supporting_passages,
                "citations": [p["passage_number"] for p in supporting_passages]
            })
        
        verification_rate = verified_count / len(sentences) if sentences else 0.0
        
        result = {
            "verified": verification_rate >= 0.8,  # 80% threshold for "verified" status
            "total_sentences": len(sentences),
            "verified_sentences": verified_count,
            "unverified_sentences": len(sentences) - verified_count,
            "verification_rate": round(verification_rate, 3),
            "sentences": verification_results,
            "min_similarity_threshold": min_similarity
        }
        
        logger.info(f" Verification complete: {verified_count}/{len(sentences)} sentences verified ({verification_rate*100:.1f}%)")
        
        if verification_rate < 0.8:
            logger.warning(f" LOW VERIFICATION RATE: {verification_rate*100:.1f}% - LLM may have hallucinated")
        
        return result
    
    @staticmethod
    def format_evidence_chain(verification_data: Dict[str, Any], include_unverified_only: bool = False) -> str:
        """Format evidence chain verification results for display.
        
        Args:
            verification_data: Output from verify_evidence_chain
            include_unverified_only: Only show unverified sentences (for debugging)
        """
        if "warning" in verification_data:
            return f"\n Warning: {verification_data['warning']}\n"
        
        output = []
        output.append(" EVIDENCE CHAIN VERIFICATION")
        
        total = verification_data["total_sentences"]
        verified = verification_data["verified_sentences"]
        unverified = verification_data["unverified_sentences"]
        rate = verification_data["verification_rate"]
        
        # Overall status
        if rate >= 0.9:
            status_icon = "✅"
            status_text = "EXCELLENT"
        elif rate >= 0.8:
            status_icon = "✓"
            status_text = "GOOD"
        elif rate >= 0.6:
            status_icon = "⚠️"
            status_text = "MODERATE"
        else:
            status_icon = "❌"
            status_text = "POOR - POSSIBLE HALLUCINATION"
        
        output.append(f"{status_icon} Verification Status: {status_text}")
        output.append(f"   Total Sentences: {total}")
        output.append(f"   Verified: {verified} ({rate*100:.1f}%)")
        output.append(f"   Unverified: {unverified}")
        output.append(f"   Threshold: {verification_data['min_similarity_threshold']}\n")
        output.append("-" * 80)
        
        # Sentence-by-sentence breakdown
        sentences = verification_data.get("sentences", [])
        
        for sent_data in sentences:
            if include_unverified_only and sent_data["verified"]:
                continue
            
            sent_num = sent_data["sentence_number"]
            sentence = sent_data["sentence"]
            is_verified = sent_data["verified"]
            citations = sent_data["citations"]
            confidence = sent_data["confidence"]
            
            if is_verified:
                icon = "✓"
                citation_str = f" [{', '.join(map(str, citations))}]"
                conf_str = f" (confidence: {confidence:.2f})"
            else:
                icon = "✗"
                citation_str = " [NO CITATION]"
                conf_str = " UNVERIFIED"
            
            output.append(f"\n{icon} Sentence {sent_num}:{citation_str}{conf_str}")
            output.append(f"   {sentence}")
            
            # Show supporting evidence for unverified claims
            if not is_verified:
                output.append("   💡 Suggestion: This claim may not be directly supported by retrieved evidence.")
        
        output.append("\n" + "="*80)
        
        if unverified > 0:
            output.append(f"\n WARNING: {unverified} sentence(s) could not be verified with retrieved evidence.")
            output.append("This may indicate:")
            output.append("  • LLM generated plausible but unsupported claims (hallucination)")
            output.append("  • Evidence is too general or implicit")
            output.append("  • Inference required beyond direct evidence")
            output.append("\n Recommendation: Review unverified sentences carefully and consult original sources.")
            output.append("="*80)
        
        return "\n".join(output)
    
    # Diagnose why low energy occurred
    def _diagnose_low_energy(self, gate_results: Dict[str, Any], gate_status: str) -> Dict[str, Any]:
        diagnosis = {
            "primary_issue": None,
            "detailed_reason": None,
            "actionable_suggestions": [],
            "alternative_approaches": [],
            "severity": "moderate"
        }
        
        if "gate1" in gate_status:
            # Gate 1: Query quality issues
            gate1_reasons = gate_results.get("gate1", {}).get("reasons", [])
            
            if any("too short" in r.lower() or "too vague" in r.lower() for r in gate1_reasons):
                diagnosis["primary_issue"] = "Query Too Vague"
                diagnosis["detailed_reason"] = "Your question lacks specific medical details needed to retrieve relevant evidence."
                diagnosis["actionable_suggestions"] = [
                    "Add specific symptoms, patient demographics (age/sex), or duration",
                    "Example: Instead of 'chest pain', try 'acute chest pain in 55-year-old male with history of hypertension'",
                    "Include relevant context: onset, severity, associated symptoms"
                ]
                diagnosis["alternative_approaches"] = [
                    "Break your question into smaller, focused sub-questions",
                    "Start with a broader search, then narrow based on findings"
                ]
                
            elif any("multiple questions" in r.lower() for r in gate1_reasons):
                diagnosis["primary_issue"] = "Multiple Questions Detected"
                diagnosis["detailed_reason"] = "Your query contains multiple distinct questions that should be asked separately."
                diagnosis["actionable_suggestions"] = [
                    "Ask one focused question at a time for better results",
                    "Prioritize your most important question first",
                    "Each medical concern deserves separate analysis"
                ]
                diagnosis["alternative_approaches"] = [
                    "Split into: 1) diagnostic question, 2) treatment question",
                    "Consider using differential diagnosis mode (--ddx flag)"
                ]
                
            elif any("broad" in r.lower() for r in gate1_reasons):
                diagnosis["primary_issue"] = "Query Too Broad"
                diagnosis["detailed_reason"] = "Your question is too general to provide specific, actionable medical guidance."
                diagnosis["actionable_suggestions"] = [
                    "Narrow your focus to a specific aspect (e.g., diagnosis, treatment, prognosis)",
                    "Add patient-specific factors that affect management",
                    "Specify which clinical context you're interested in"
                ]
                diagnosis["alternative_approaches"] = [
                    "Use differential diagnosis mode for symptom clusters",
                    "Focus on one decision point at a time"
                ]
            else:
                diagnosis["primary_issue"] = "Query Unclear"
                diagnosis["detailed_reason"] = "The query structure or phrasing makes it difficult to understand your medical question."
                diagnosis["actionable_suggestions"] = [
                    "Rephrase using standard medical terminology",
                    "Structure as: Patient presentation → Clinical question",
                    "Be explicit about what you want to know"
                ]
                
        elif "gate2" in gate_status:
            # Gate 2: Retrieval quality issues
            gate2_reasons = gate_results.get("gate2", {}).get("reasons", [])
            
            if any("insufficient passages" in r.lower() or "too few" in r.lower() for r in gate2_reasons):
                diagnosis["primary_issue"] = "Insufficient Evidence in Database"
                diagnosis["detailed_reason"] = "The knowledge base doesn't contain enough relevant information for this query."
                diagnosis["severity"] = "high"
                diagnosis["actionable_suggestions"] = [
                    "Try alternative medical terms or synonyms",
                    "Broaden your search slightly (e.g., disease category instead of specific condition)",
                    "Check if this is a rare condition that may require specialized resources"
                ]
                diagnosis["alternative_approaches"] = [
                    "This may require consulting specialized medical databases",
                    "Consider expert consultation for rare or emerging conditions",
                    "Try searching for related conditions with similar presentations"
                ]
                
            elif any("short passages" in r.lower() for r in gate2_reasons):
                diagnosis["primary_issue"] = "Low-Quality Evidence Retrieved"
                diagnosis["detailed_reason"] = "Retrieved passages are too brief or incomplete to provide confident answers."
                diagnosis["actionable_suggestions"] = [
                    "Add more clinical context to help retrieve comprehensive passages",
                    "Try different search terms that might match longer clinical discussions"
                ]
                
            elif any("duplicat" in r.lower() for r in gate2_reasons):
                diagnosis["primary_issue"] = "Repetitive Evidence"
                diagnosis["detailed_reason"] = "Retrieved evidence is highly redundant without sufficient diversity."
                diagnosis["actionable_suggestions"] = [
                    "Evidence base may have limited coverage for this topic",
                    "Try searching related aspects of the condition"
                ]
                
        elif "gate3" in gate_status:
            # Gate 3: Evidence consistency issues
            gate3_reasons = gate_results.get("gate3", {}).get("reasons", [])
            
            if any("only" in r.lower() and "passage" in r.lower() for r in gate3_reasons):
                diagnosis["primary_issue"] = "Single Source Evidence"
                diagnosis["detailed_reason"] = "Only one passage meets quality threshold - insufficient for confident medical guidance."
                diagnosis["severity"] = "high"
                diagnosis["actionable_suggestions"] = [
                    "⚠️ CRITICAL: Single-source medical advice is unreliable",
                    "Seek multiple evidence sources or expert consultation",
                    "This finding may indicate a knowledge gap in the database"
                ]
                diagnosis["alternative_approaches"] = [
                    "Consult primary medical literature or clinical guidelines",
                    "Consider this a preliminary finding requiring verification"
                ]
                
            elif any("low average similarity" in r.lower() for r in gate3_reasons):
                diagnosis["primary_issue"] = "Weak Evidence Match"
                diagnosis["detailed_reason"] = "Retrieved evidence has low semantic relevance to your query."
                diagnosis["actionable_suggestions"] = [
                    "Rephrase using terminology more likely to appear in medical literature",
                    "Your question might be at the edge of available evidence",
                    "Try breaking into more specific sub-questions"
                ]
                
            elif any("variance" in r.lower() or "inconsistent" in r.lower() for r in gate3_reasons):
                diagnosis["primary_issue"] = "Conflicting Evidence"
                diagnosis["detailed_reason"] = "Evidence sources show high variability or disagreement."
                diagnosis["severity"] = "high"
                diagnosis["actionable_suggestions"] = [
                    "⚠️ IMPORTANT: Evidence conflict detected - requires expert interpretation",
                    "Use --verify-evidence flag to see which claims are unsupported",
                    "Enable contradiction detection to understand disagreements"
                ]
                diagnosis["alternative_approaches"] = [
                    "This may indicate evolving medical consensus",
                    "Different guidelines or patient populations may explain variance",
                    "Expert consultation recommended for nuanced interpretation"
                ]
        
        # Add severity indicator
        if diagnosis["severity"] == "high":
            diagnosis["detailed_reason"] = "🔴 " + diagnosis["detailed_reason"]
        
        return diagnosis
    
    # gracefully say no when needed
    @staticmethod
    def format_graceful_refusal(result: Dict[str, Any]) -> str:
        output = []
        output.append("❌ QUERY REJECTED - Unable to Provide Confident Answer")
                
        # Basic rejection info
        gate_status = result.get("gate_status", "unknown")
        output.append(f"Rejected at: {gate_status.replace('_', ' ').replace('rejected at ', '').upper()}")
        output.append(f"Compute saved: {result.get('compute_saved', 'N/A')}\n")
        output.append("-" * 80)
        
        # Show diagnosis if available
        diagnosis = result.get("diagnosis")
        if diagnosis:
            issue = diagnosis.get("primary_issue", "Unknown Issue")
            reason = diagnosis.get("detailed_reason", "Unable to determine specific reason")
            
            output.append(f"\n Primary Issue: {issue}")
            output.append(f"   {reason}\n")
            output.append("-" * 80)
            
            # Actionable suggestions
            suggestions = diagnosis.get("actionable_suggestions", [])
            if suggestions:
                output.append("\n ACTIONABLE SUGGESTIONS:")
                for i, suggestion in enumerate(suggestions, 1):
                    output.append(f"   {i}. {suggestion}")
                output.append("")
            
            # Alternative approaches
            alternatives = diagnosis.get("alternative_approaches", [])
            if alternatives:
                output.append(" ALTERNATIVE APPROACHES:")
                for alt in alternatives:
                    output.append(f"   • {alt}")
                output.append("")
        
        # Show gate scores breakdown
        gates = result.get("gates", {})
        if gates:
            output.append("GATE SCORES BREAKDOWN:")
            for gate_name, gate_data in gates.items():
                status = "✓ PASS" if gate_data.get("passed") else "✗ FAIL"
                score = gate_data.get("energy_score", 0)
                threshold = gate_data.get("threshold", 0)
                output.append(f"\n  {gate_name.upper()}: {score:.2f} / {threshold:.2f} {status}")
                
                reasons = gate_data.get("reasons", [])
                for reason in reasons:
                    output.append(f"    • {reason}")
        
        # Show partial evidence if available
        partial_evidence = result.get("partial_evidence", [])
        if partial_evidence:
            output.append("\n" + "-" * 80)
            output.append("PARTIAL EVIDENCE FOUND (for reference only):")
            output.append("Note: This evidence did not meet confidence thresholds\n")
            
            for i, (passage, metadata, score) in enumerate(partial_evidence[:3], 1):
                source = metadata.get("subject_name", "Unknown")
                topic = metadata.get("topic_name", "")
                output.append(f"  [{i}] {source} - {topic} (similarity: {score:.2f})")
                output.append(f"      {passage[:200]}...")
                output.append("")
        
        output.append("=" * 80)
        output.append(" RECOMMENDATION: Review suggestions above and reformulate your query")
        output.append("=" * 80)
        
        return "\n".join(output)
    
    # Detect contradictions and disagreements between sources
    def detect_contradictions(self, filtered_passages: List[Tuple[str, Dict, float]], 
                             user_query: str) -> Dict[str, Any]:
        if len(filtered_passages) < 2:
            return {"has_contradictions": False, "conflicts": []}
        
        logger.info("Analyzing sources for contradictions...")
        
        # Prepare evidence for LLM analysis
        evidence_list = []
        for i, (passage, metadata, score) in enumerate(filtered_passages[:10], 1):
            source_info = f"Source {i}"
            if metadata.get("subject_name"):
                source_info += f" ({metadata.get('subject_name', 'Unknown')})"
            if metadata.get("topic_name"):
                source_info += f" - {metadata.get('topic_name', '')}"
            
            evidence_list.append(f"{source_info}:\n{passage}")
        
        evidence_text = "\n\n".join(evidence_list)
        
        # Use LLM to detect contradictions
        system_prompt = """You are a medical evidence analysis expert specializing in detecting contradictions.

TASK: Analyze medical evidence sources for contradictions, disagreements, or conflicting claims.

CRITICAL RULES:
1. Compare specific claims (numbers, recommendations, dosages, guidelines)
2. Identify substantive conflicts - not just different phrasings
3. For each conflict, quote the exact conflicting statements
4. Suggest possible reasons for disagreements (population differences, guideline updates, risk stratification)
5. If no meaningful contradictions exist, clearly state that

RESPONSE FORMAT (JSON):
{
  "has_contradictions": true,
  "conflicts": [
    {
      "topic": "Blood pressure targets",
      "source_1": {
        "claim": "Target BP <130/80",
        "source_id": "Source 1 (AHA 2024)"
      },
      "source_2": {
        "claim": "Target BP <140/90", 
        "source_id": "Source 2 (ESC 2023)"
      },
      "severity": "moderate",
      "possible_reasons": [
        "Different patient populations (US vs EU guidelines)",
        "Different risk stratification approaches",
        "Updated evidence in newer guideline"
      ],
      "recommendation": "Consider patient-specific factors including age, comorbidities, and risk profile"
    }
  ],
  "overall_assessment": "Moderate conflicts found - guidelines differ on specific thresholds but agree on treatment principles"
}

SEVERITY LEVELS:
- "critical": Directly opposite recommendations (do vs don't)
- "moderate": Numerical differences or timing variations
- "minor": Different approaches to same goal"""
        
        user_prompt = f"""Query: {user_query}

Analyze these evidence sources for contradictions:

{evidence_text}

Return valid JSON only, no markdown."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                contradiction_data = json.loads(json_match.group())
            else:
                contradiction_data = json.loads(response_text)
            
            logger.info(f"Contradiction analysis complete: {contradiction_data.get('has_contradictions', False)}")
            return contradiction_data
            
        except Exception as e:
            logger.error(f"Failed to analyze contradictions: {e}")
            return {
                "has_contradictions": False,
                "conflicts": [],
                "error": str(e)
            }
    
    @staticmethod
    def format_contradictions(contradiction_data: Dict[str, Any]) -> str:
        if not contradiction_data.get("has_contradictions", False):
            return "\n✓ No significant contradictions detected between sources\n"
        
        output = []
        output.append("\n" + "⚠️ "*40)
        output.append("⚠️  CONFLICTING EVIDENCE DETECTED")
        output.append("⚠️ "*40 + "\n")
        
        conflicts = contradiction_data.get("conflicts", [])
        for i, conflict in enumerate(conflicts, 1):
            topic = conflict.get("topic", "Unknown topic")
            severity = conflict.get("severity", "unknown").upper()
            
            # Severity icon
            severity_icon = {
                "CRITICAL": "🔴",
                "MODERATE": "🟡", 
                "MINOR": "🟢"
            }.get(severity, "⚠️")
            
            output.append(f"{severity_icon} Conflict #{i}: {topic} [{severity} SEVERITY]")
            output.append("-" * 80)
            
            # Source 1
            source1 = conflict.get("source_1", {})
            output.append(f"\n{source1.get('source_id', 'Source 1')}:")
            output.append(f'  "{source1.get("claim", "N/A")}"')
            
            # Source 2
            source2 = conflict.get("source_2", {})
            output.append(f"\n{source2.get('source_id', 'Source 2')}:")
            output.append(f'  "{source2.get("claim", "N/A")}"')
            
            # Possible reasons
            reasons = conflict.get("possible_reasons", [])
            if reasons:
                output.append("\nPossible reasons for disagreement:")
                for reason in reasons:
                    output.append(f"  • {reason}")
            
            # Recommendation
            recommendation = conflict.get("recommendation", "")
            if recommendation:
                output.append(f"\n💡 Recommendation: {recommendation}")
            
            output.append("\n")
        
        # Overall assessment
        overall = contradiction_data.get("overall_assessment", "")
        if overall:
            output.append("-" * 80)
            output.append(f"Overall Assessment: {overall}")
            output.append("-" * 80)
        
        return "\n".join(output)
    
    def answer_query(self, user_query: str, top_k: int = 10, threshold: float = 0.5, 
                     use_llm: bool = True, detect_conflicts: bool = True,
                     enable_gates: bool = True,
                     gate_thresholds: Dict[str, float] = None,
                     verify_chain: bool = False) -> Dict[str, Any]:
        logger.info("="*80)
        logger.info(f"Query: {user_query}")
        logger.info("="*80)
        
        # Set default gate thresholds
        if gate_thresholds is None:
            gate_thresholds = {
                "gate1": 0.7,   # Query quality
                "gate2": 0.6,   # Retrieval quality  
                "gate3": 0.65   # Evidence consistency
            }
        
        gate_results = {}
        
        if enable_gates:
            gate1_result = self.gate_1_query_quality(user_query, gate_thresholds["gate1"])
            gate_results["gate1"] = gate1_result
            
            if not gate1_result["passed"]:
                logger.warning("Gate 1 REJECTED - Stopping pipeline early")
                diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate1")
                return {
                    "query": user_query,
                    "gate_status": "rejected_at_gate1",
                    "gates": gate_results,
                    "retrieved_count": 0,
                    "ranked_passages": [],
                    "filtered_passages": [],
                    "partial_evidence": [],  # No retrieval done yet
                    "context": None,
                    "answer": None,
                    "rejection_reason": f"Query quality too low: {', '.join(gate1_result['reasons'])}",
                    "suggestion": "Please rephrase your question to be more specific and clear.",
                    "compute_saved": "Saved: retrieval + reranking + LLM",
                    "confidence": "rejected",
                    "diagnosis": diagnosis
                }
        
        # Step 1: Retrieve
        passages, metadatas = self.retrieve_top_k(user_query, top_k)
        logger.info(f"[1/5] Retrieved {len(passages)} passages")
        
        if enable_gates:
            gate2_result = self.gate_2_retrieval_quality(passages, metadatas, gate_thresholds["gate2"])
            gate_results["gate2"] = gate2_result
            
            if not gate2_result["passed"]:
                logger.warning(" Gate 2 REJECTED - Stopping pipeline early")
                diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate2")
                # Provide partial evidence (top 3 retrieved passages even if low quality)
                partial_evidence = [(passages[i], metadatas[i], 0.0) for i in range(min(3, len(passages)))]
                return {
                    "query": user_query,
                    "gate_status": "rejected_at_gate2",
                    "gates": gate_results,
                    "retrieved_count": len(passages),
                    "ranked_passages": [],
                    "filtered_passages": [],
                    "partial_evidence": partial_evidence,
                    "context": None,
                    "answer": None,
                    "rejection_reason": f"Retrieval quality too low: {', '.join(gate2_result['reasons'])}",
                    "suggestion": "Insufficient evidence found. This may require specialized medical consultation.",
                    "compute_saved": "Saved: reranking + LLM",
                    "confidence": "rejected",
                    "diagnosis": diagnosis
                }
        
        # Step 2: Rerank
        ranked = self.rerank_passages(user_query, passages, metadatas)
        logger.info(f"[2/5] Ranked {len(ranked)} passages")
        
        # Step 3: Filter
        filtered = self.filter_by_threshold(ranked, threshold)
        logger.info(f"[3/5] Filtered to {len(filtered)} high-confidence passages")
        
        if enable_gates:
            gate3_result = self.gate_3_evidence_consistency(filtered, gate_thresholds["gate3"])
            gate_results["gate3"] = gate3_result
            
            if not gate3_result["passed"]:
                logger.warning(" Gate 3 REJECTED - Stopping pipeline early")
                diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate3")
                context = self.prepare_for_llm(filtered) if filtered else "No evidence passed threshold"
                # Provide partial evidence (whatever passed filtering)
                partial_evidence = filtered[:3] if filtered else []
                return {
                    "query": user_query,
                    "gate_status": "rejected_at_gate3",
                    "gates": gate_results,
                    "retrieved_count": len(passages),
                    "ranked_passages": ranked,
                    "filtered_passages": filtered,
                    "partial_evidence": partial_evidence,
                    "context": context,
                    "answer": None,
                    "rejection_reason": f"Evidence consistency too low: {', '.join(gate3_result['reasons'])}",
                    "suggestion": "Evidence is too weak or inconsistent to provide a confident answer. This requires personalized medical advice.",
                    "compute_saved": "Saved: LLM call",
                    "confidence": "rejected",
                    "diagnosis": diagnosis,
                    "evidence_metadata": {
                        "num_sources": len(filtered),
                        "avg_similarity": gate3_result.get("metrics", {}).get("avg_similarity", 0.0),
                        "topics": list(set(m.get("topic_name", "") for _, m, _ in filtered if m.get("topic_name")))
                    }
                }
        
        # All gates passed - continue with normal pipeline
        if enable_gates:
            logger.info("All gates PASSED - Proceeding to LLM")
        
        # NEW: Check if we have enough evidence for quality control
        confidence_level = "low"
        warning_message = None
        
        if len(filtered) < 2:
            logger.warning(f"Only {len(filtered)} passages passed threshold for query: {user_query}")
            warning_message = "Limited evidence available - answer confidence is low"
            confidence_level = "low"
        elif len(filtered) >= 5:
            logger.info(f"Strong evidence base: {len(filtered)} passages for query: {user_query}")
            confidence_level = "high"
        else:
            confidence_level = "medium"
        
        # Step 4: LLM or context only
        context = self.prepare_for_llm(filtered)
        
        # NEW: Detect contradictions if requested
        contradiction_data = None
        if detect_conflicts and len(filtered) >= 2:
            logger.info("[4/5] Checking for contradictions...")
            contradiction_data = self.detect_contradictions(filtered, user_query)
        
        # NEW: Calculate evidence metadata
        avg_similarity = sum(p[2] for p in filtered) / len(filtered) if filtered else 0.0
        topics = list(set(m.get("topic_name", "") for _, m, _ in filtered if m.get("topic_name")))
        
        result = {
            "query": user_query,
            "gate_status": "all_gates_passed" if enable_gates else "gates_disabled",
            "gates": gate_results if enable_gates else None,
            "retrieved_count": len(passages),
            "ranked_passages": ranked,
            "filtered_passages": filtered,
            "context": context,
            "answer": None,
            "follow_up_questions": [],  # Initialize as empty list
            "answer_time": None,
            "confidence": confidence_level,
            "warning": warning_message,
            "contradictions": contradiction_data,
            "evidence_metadata": {
                "num_sources": len(filtered),
                "avg_similarity": round(avg_similarity, 3),
                "topics": topics,
            }
        }
        
        if use_llm and filtered:
            logger.info(f"[5/6] Querying LLM...")
            start_time = datetime.now()
            answer, follow_up_questions = self.query_llm(user_query, context)
            end_time = datetime.now()
            result["answer"] = answer
            result["follow_up_questions"] = follow_up_questions
            result["answer_time"] = (end_time - start_time).total_seconds()
            logger.info(f"[Done] Answer generated in {result['answer_time']:.2f}s with {len(follow_up_questions)} follow-up questions")
            
            # Step 6: Verify evidence chain (if requested and answer exists)
            if verify_chain and answer:
                logger.info("[6/6] Verifying evidence chain...")
                evidence_chain = self.verify_evidence_chain(answer, filtered, min_similarity=0.6)
                result["evidence_chain"] = evidence_chain
                
                # Warn if verification rate is low
                if evidence_chain["verification_rate"] < 0.8:
                    logger.warning(
                        f" LOW VERIFICATION RATE: {evidence_chain['verification_rate']*100:.1f}% - "
                        f"{evidence_chain['unverified_sentences']} sentence(s) may be hallucinated"
                    )
        else:
            logger.info(f"[5/6] Skipping LLM (use_llm={use_llm}, filtered={len(filtered)>0})")
        
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

        CRITICAL RULES:
        1. ONLY use information from the provided evidence - do not use external medical knowledge
        2. If evidence is conflicting, acknowledge and explain the conflict
        3. If evidence is insufficient, explicitly state: "I don't have enough evidence to answer confidently. 🐱💤"
        4. Always cite which evidence you used (e.g., "[Evidence 1]", "[Evidence 2]")
        5. Use clear medical terminology but explain complex terms for general audience
        6. Structure answers for clarity: start with direct answer, then support with evidence

        RESPONSE FORMAT:
        - Opening: Direct, evidence-based answer to the question
        - Body: Detailed explanation with specific evidence citations
        - Supporting Citations: List which evidence items support each claim
        - Confidence Level: End with assessment (High/Medium/Low) based on evidence quality and consistency
        - Disclaimers: Note if evidence is limited or conflicting
        - Follow-up Questions: Suggest 3-5 relevant follow-up questions the user might ask based on this topic

        IMPORTANT: Do NOT invent or assume information not explicitly in the evidence."""
        
        user_prompt = f"""Medical Question: {user_query}

Available Evidence:
{context}

Please provide a comprehensive, evidence-based answer following the critical rules above.
Make sure to cite specific evidence items and provide a confidence assessment.

At the end, add a section:
---
**Suggested Follow-up Questions:**
1. [relevant follow-up question based on answer]
2. [another related question]
3. [deeper dive question]
4. [practical application question]
5. [related condition/treatment question]"""
        
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
            self.cache.set(cache_key, answer, expire=self.cache_ttl)
        
        # Extract follow-up questions if present
        follow_up_questions = []
        if "Suggested Follow-up Questions:" in answer or "Follow-up Questions:" in answer:
            import re
            # Split on the follow-up questions marker
            parts = re.split(r'\*\*Suggested Follow-up Questions:\*\*|Follow-up Questions:', answer)
            if len(parts) > 1:
                questions_section = parts[-1]
                # Extract numbered questions
                question_matches = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', questions_section, re.DOTALL)
                follow_up_questions = [q.strip().strip('[]').strip() for q in question_matches if q.strip()]
        
        return answer, follow_up_questions
    
    def generate_differential_diagnosis(self, user_query: str, top_k: int = 10, 
                                       threshold: float = 0.5, num_diagnoses: int = 5) -> Dict[str, Any]:
        """Generate ranked differential diagnoses with confidence explanations.
        
        Args:
            user_query: Clinical presentation/symptoms
            top_k: Top passages to retrieve
            threshold: Similarity threshold for filtering
            num_diagnoses: Number of diagnoses to generate
            
        Returns:
            Dictionary with ranked diagnoses and confidence explanations
        """
        logger.info(f"Generating differential diagnosis for: {user_query}")
        
        # Retrieve and rank evidence
        passages, metadatas = self.retrieve_top_k(user_query, top_k)
        ranked = self.rerank_passages(user_query, passages, metadatas)
        filtered = self.filter_by_threshold(ranked, threshold)
        context = self.prepare_for_llm(filtered)
        
        # Generate differential diagnoses via LLM
        system_prompt = """You are a medical differential diagnosis expert.

TASK: Generate ranked differential diagnoses based on clinical presentation.

CRITICAL RULES:
1. ONLY use information from provided evidence - no external knowledge
2. Rank diagnoses by likelihood (% match confidence)
3. For each diagnosis include:
   - Condition name
   - Match percentage (0-100%)
   - Key matching symptoms
   - Number of supporting evidence sources
4. If evidence is limited, explicitly state confidence limitations
5. Always explain WHICH symptoms led to WHICH diagnoses

RESPONSE FORMAT (JSON):
{
  "diagnoses": [
    {
      "rank": 1,
      "condition": "Pneumonia",
      "match_percentage": 85,
      "key_symptoms": ["fever", "productive cough", "pleuritic chest pain"],
      "evidence_count": 3,
      "evidence_sources": ["[Evidence 1]", "[Evidence 3]", "[Evidence 5]"],
      "reasoning": "Classic presentation with fever + cough + chest pain in 3 peer-reviewed sources"
    }
  ],
  "overall_confidence": "high",
  "confidence_reasoning": "Multiple high-quality sources with consistent diagnostic criteria"
}"""
        
        user_prompt = f"""Clinical Presentation: {user_query}

Available Evidence:
{context}

Generate a ranked differential diagnosis (top {num_diagnoses} conditions).
Return response as valid JSON only, no markdown."""
        
        response = self.llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ddx_data = json.loads(json_match.group())
            else:
                ddx_data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM differential diagnosis response")
            ddx_data = {"diagnoses": [], "overall_confidence": "unknown"}
        
        # Calculate metadata
        avg_similarity = sum(p[2] for p in filtered) / len(filtered) if filtered else 0.0
        topics = list(set(m.get("topic_name", "") for _, m, _ in filtered if m.get("topic_name")))
        
        result = {
            "query": user_query,
            "diagnoses": ddx_data.get("diagnoses", []),
            "overall_confidence": ddx_data.get("overall_confidence", "unknown"),
            "confidence_explanation": self._build_confidence_explanation(
                filtered, avg_similarity, ddx_data.get("overall_confidence", "unknown")
            ),
            "evidence_metadata": {
                "num_sources": len(filtered),
                "avg_similarity": round(avg_similarity, 3),
                "topics": topics,
            }
        }
        
        return result
    
    # Build confidence explanation based on evidence metrics
    def _build_confidence_explanation(self, filtered_passages: List[Tuple[str, Dict, float]], 
                                     avg_similarity: float, confidence_level: str) -> Dict[str, Any]:
        explanation = {
            "level": confidence_level,
            "reasoning": [],
            "concerns": []
        }
        
        # Positive factors
        if len(filtered_passages) >= 5:
            explanation["reasoning"].append(f"✓ Strong evidence base: {len(filtered_passages)} high-quality sources")
        elif len(filtered_passages) >= 3:
            explanation["reasoning"].append(f"✓ Moderate evidence: {len(filtered_passages)} sources available")
        else:
            explanation["reasoning"].append(f"⚠ Limited evidence: only {len(filtered_passages)} sources")
        
        if avg_similarity >= 0.75:
            explanation["reasoning"].append("✓ High semantic match (avg similarity: {:.2f}) between query and evidence".format(avg_similarity))
        elif avg_similarity >= 0.5:
            explanation["reasoning"].append("⚠ Moderate semantic match (avg similarity: {:.2f})".format(avg_similarity))
        else:
            explanation["concerns"].append("✗ Low semantic match - evidence may not fully address query")
        
        # Evidence quality checks
        peer_reviewed_count = sum(1 for _, m, _ in filtered_passages if m.get("source_type") == "peer-reviewed")
        if peer_reviewed_count > 0:
            explanation["reasoning"].append(f"✓ {peer_reviewed_count} peer-reviewed sources (high-quality evidence)")
        
        # Evidence recency
        recent_sources = sum(1 for _, m, _ in filtered_passages if m.get("year", 0) >= 2020)
        if recent_sources > 0:
            explanation["reasoning"].append(f"✓ {recent_sources} sources published within last 4 years (recent evidence)")
        else:
            explanation["concerns"].append("✗ Evidence may be outdated - check publication dates")
        
        # Confidence assessment
        if len(filtered_passages) < 2:
            explanation["concerns"].append("✗ CRITICAL: Insufficient evidence - diagnosis may be unreliable")
        
        return explanation
    
    @staticmethod
    def format_differential_diagnosis(ddx_result: Dict[str, Any]) -> str:
        output = []
        output.append("\n" + "="*80)
        output.append(f"DIFFERENTIAL DIAGNOSIS: {ddx_result['query']}")
        output.append("="*80)
        
        # Print diagnoses
        diagnoses = ddx_result.get("diagnoses", [])
        if diagnoses:
            for dx in diagnoses:
                rank = dx.get("rank", "?")
                condition = dx.get("condition", "Unknown")
                match_pct = dx.get("match_percentage", 0)
                key_symptoms = dx.get("key_symptoms", [])
                evidence_count = dx.get("evidence_count", 0)
                reasoning = dx.get("reasoning", "")
                
                # Color-code by match percentage
                if match_pct >= 75:
                    confidence_icon = "✓✓"
                elif match_pct >= 50:
                    confidence_icon = "✓"
                else:
                    confidence_icon = "⚠"
                
                output.append(f"\n{rank}. {condition} ({match_pct}% match) {confidence_icon}")
                output.append(f"   Evidence: [{evidence_count} sources]")
                output.append(f"   Key symptoms match: {', '.join(key_symptoms)}")
                if reasoning:
                    output.append(f"   Reasoning: {reasoning}")
        else:
            output.append("\nNo diagnoses generated")
        
        # Print confidence explanation
        output.append("\n" + "-"*80)
        output.append("CONFIDENCE EXPLANATION")
        output.append("-"*80)
        
        confidence_explain = ddx_result.get("confidence_explanation", {})
        confidence_level = confidence_explain.get("level", "unknown")
        output.append(f"\nOverall Confidence: {confidence_level.upper()}")
        
        # Positive reasoning
        reasoning = confidence_explain.get("reasoning", [])
        if reasoning:
            output.append("\nWhy we're confident:")
            for reason in reasoning:
                output.append(f"  {reason}")
        
        # Concerns
        concerns = confidence_explain.get("concerns", [])
        if concerns:
            output.append("\nLimitations:")
            for concern in concerns:
                output.append(f"  {concern}")
        
        # Evidence metadata
        output.append("\n" + "-"*80)
        output.append("EVIDENCE SUMMARY")
        output.append("-"*80)
        evidence = ddx_result.get("evidence_metadata", {})
        output.append(f"Total sources: {evidence.get('num_sources', 0)}")
        output.append(f"Avg similarity score: {evidence.get('avg_similarity', 0):.3f}")
        topics = evidence.get("topics", [])
        if topics:
            output.append(f"Topics covered: {', '.join(topics)}")
        
        output.append("="*80 + "\n")
        
        return "\n".join(output)


def main():
    """CLI entrypoint: ask a question and talk to the LLM."""
    parser = argparse.ArgumentParser(description="Query the EBM retrieval + LLM pipeline")
    parser.add_argument("query", nargs="?", help="User question or clinical presentation")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k passages to retrieve")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold to keep passages")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM call (retrieval + ranking only)")
    parser.add_argument("--ddx", action="store_true", help="Generate differential diagnosis instead of direct answer")
    parser.add_argument("--no-conflict-check", action="store_true", help="Skip contradiction detection")
    parser.add_argument("--no-gates", action="store_true", help="Disable multi-level energy gating")
    parser.add_argument("--verify-evidence", action="store_true", help="Verify each sentence in answer is backed by evidence")
    args = parser.parse_args()

    # If query not provided as an argument, prompt the user interactively
    if not args.query:
        try:
            args.query = input("Enter your medical question: ").strip()
        except EOFError:
            args.query = None

    if not args.query:
        logger.error("No query provided. Please pass a query argument or type one when prompted.")
        return

    use_llm = not args.no_llm

    pipeline = RetrievalPipeline(
        chroma_path=str(CHROMA_PATH),
        collection_name=CHROMA_COLLECTION_NAME,
        model_name=MODEL_NAME,
        checkpoint_path=str(RANKING_SCORER_CKPT),
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_cache=True,
    )

    # Generate differential diagnosis or standard answer
    if args.ddx:
        result = pipeline.generate_differential_diagnosis(
            user_query=args.query,
            top_k=args.top_k,
            threshold=args.threshold,
            num_diagnoses=5
        )
        print(RetrievalPipeline.format_differential_diagnosis(result))
    else:
        result = pipeline.answer_query(
            user_query=args.query,
            top_k=args.top_k,
            threshold=args.threshold,
            use_llm=use_llm,
            detect_conflicts=not args.no_conflict_check,
            enable_gates=not args.no_gates,
            verify_chain=args.verify_evidence,
        )

        # Check if rejected by gates - use graceful refusal format
        if result.get("gate_status") and "rejected" in result["gate_status"]:
            print(RetrievalPipeline.format_graceful_refusal(result))
            return

        # Display contradictions if detected
        if result.get("contradictions") and result["contradictions"].get("has_contradictions"):
            print(RetrievalPipeline.format_contradictions(result["contradictions"]))
        
        # Display gate status if enabled
        if result.get("gates"):
            print("\n" + "✅"*40)
            print("ALL ENERGY GATES PASSED")
            print("✅"*40)
            for gate_name, gate_data in result["gates"].items():
                print(f"  {gate_name.upper()}: {gate_data['energy_score']:.2f} ✓")
            print()

        print("\n" + "="*80)
        print("ANSWER")
        print("="*80)
        if use_llm and result.get("answer"):
            print(result["answer"])
        else:
            print("LLM skipped; context prepared.")
        
        # Display evidence chain verification if performed
        if result.get("evidence_chain"):
            print(RetrievalPipeline.format_evidence_chain(result["evidence_chain"]))

        print("\n" + "="*80)
        print("EVIDENCE")
        print("="*80)
        print(result.get("context", ""))



if __name__ == "__main__":
    main()