import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import logging
logger = logging.getLogger("retriever")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

import os
import argparse
from typing import Any, Dict, List, Tuple
from chromadb import PersistentClient
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
import hashlib
import json
from diskcache import Cache
from datetime import datetime
import sys
from pathlib import Path
import requests

# Placeholder for extract_concepts if not imported elsewhere
def extract_concepts(text):
    # TODO: Replace with actual concept extraction logic
    # For now, return an empty set to avoid errors
    return set()

# Placeholder for PubMed search functions
def pubmed_search(query: str, max_results: int = 5, years: int = 2) -> List[str]:
    """Placeholder PubMed search function. Returns empty list for now."""
    logger.warning("PubMed search not implemented - returning empty results")
    return []

def pubmed_fetch_abstracts(pmids: List[str]) -> List[Dict[str, Any]]:
    """Placeholder PubMed abstract fetch function. Returns empty list for now."""
    logger.warning("PubMed abstract fetch not implemented - returning empty results")
    return []

# Import config constants
from config import (
    MODEL_NAME, CHROMA_PATH, CHROMA_COLLECTION_NAME, RANKING_SCORER_CKPT, VERIFICATION_SCORER_CKPT, CACHE_DIR, THRESHOLDS
)

# Setup logger
logger = logging.getLogger("retriever")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Placeholder for extract_concepts if not imported elsewhere
def extract_concepts(text):
    # TODO: Replace with actual concept extraction logic
    # For now, return an empty set to avoid errors
    return set()


# ...existing code replaced by improved version provided by user...

# Ranking model (from train.ipynb)
class RankingScorer(nn.Module):
    def __init__(self, encoder, embedding_dim=None):
        super().__init__()
        if embedding_dim is None:
            from config import THRESHOLDS
            embedding_dim = THRESHOLDS["embedding_dim"]
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
                 verification_checkpoint_path: str = None,
                 device: str = None,
                 llm_api_key: str = None,
                 enable_cache: bool = True,
                 cache_dir: str = None):
        # Load environment variables from .env file
        load_dotenv()
        
        # Use config MODEL_NAME by default
        model_name = model_name or config.MODEL_NAME
        # Use config defaults if not provided
        chroma_path = chroma_path or str(config.CHROMA_PATH)
        collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        checkpoint_path = checkpoint_path or str(config.RANKING_SCORER_CKPT)
        verification_checkpoint_path = verification_checkpoint_path or str(config.VERIFICATION_SCORER_CKPT)
        cache_dir = cache_dir or str(config.CACHE_DIR)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.enable_cache = enable_cache
        
        # Initialize cache (retrieval + LLM only)
        if enable_cache:
            # Cache with size limit (1GB) and LRU eviction policy
            self.cache = Cache(cache_dir, 
                               size_limit=config.THRESHOLDS["cache_size_limit"], 
                               eviction_policy='least-recently-used')
            # Set TTL for cache entries (7 days in seconds)
            self.cache_ttl = config.THRESHOLDS["cache_ttl_seconds"]
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
        self.ranking_model = RankingScorer(encoder, embedding_dim=config.THRESHOLDS["embedding_dim"]).to(device)
        # Use weights_only=True to avoid loading arbitrary pickled objects
        self.ranking_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        self.ranking_model.eval()
        logger.info(f"Loaded ranking model from {checkpoint_path}")
        
        # Load verification scorer (separate model for verification)
        self.verification_model = RankingScorer(encoder, embedding_dim=config.THRESHOLDS["embedding_dim"]).to(device)
        self.verification_model.load_state_dict(torch.load(verification_checkpoint_path, map_location=device, weights_only=True))
        self.verification_model.eval()
        logger.info(f"Loaded verification model from {verification_checkpoint_path}")
        
        # Initialize NLI pipeline for evidence verification
        nli_device = 0 if torch.cuda.is_available() else -1
        self.nli_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=nli_device)
        logger.info("Loaded NLI model for evidence verification")
        
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
        
    def _chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Chunk text into smaller pieces for ranking."""
        if chunk_size is None:
            chunk_size = config.THRESHOLDS["text_chunk_size"]
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
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
                       max_len: int = None, batch_size: int = None) -> List[Tuple[str, Dict, float]]:
        """Rerank passages using the trained ranking model."""
        if max_len is None:
            max_len = config.THRESHOLDS["rerank_max_len"]
        if batch_size is None:
            batch_size = config.THRESHOLDS["rerank_batch_size"]
        """Rerank passages using the trained ranking model."""
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
        
        return sorted(zip(passages, metadatas, scores), key=lambda x: x[2], reverse=True)
    
    def compute_energy_scores(self, passages: List[str], max_len: int = None, 
                             batch_size: int = None) -> List[float]:
        if max_len is None:
            max_len = config.THRESHOLDS["rerank_max_len"]
        if batch_size is None:
            batch_size = config.THRESHOLDS["rerank_batch_size"]
        
        # Use a reference "good evidence" query for energy computation
        reference_query = "clinical evidence medical study research"
        
        energies = []
        
        device_type = "cuda" 
        with torch.no_grad():
            # Encode reference query
            enc_ref = self.tokenizer(
                reference_query,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            # Ensure tensors are on the correct device
            enc_ref = {k: v.to(self.device) for k, v in enc_ref.items()}
            ref_embedding = self.verification_model(enc_ref)  # [1, embedding_dim]
            
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
                # Ensure tensors are on the correct device
                enc_passages = {k: v.to(self.device) for k, v in enc_passages.items()}
                passage_embeddings = self.verification_model(enc_passages)  # [batch_size, embedding_dim]
                
                # Energy = 1 - similarity (lower energy = more evidence-like)
                sims = F.cosine_similarity(ref_embedding, passage_embeddings)  # [batch_size]
                batch_energies = 1.0 - sims  # Raw energy scores
                
                # Normalize energy using sigmoid(-energy) - converts to 0-1 scale where lower energy = better
                normalized_energies = torch.sigmoid(-batch_energies)
                batch_energies_normalized = normalized_energies.detach().cpu().tolist()
                
                energies.extend(batch_energies_normalized)
        
        return energies
    
    def filter_by_threshold(self, ranked_passages: List[Tuple[str, Dict, float]], 
                           threshold: float = None) -> List[Tuple[str, Dict, float]]:
        if threshold is None:
            threshold = config.THRESHOLDS["similarity_threshold"]
        
        filtered = [p for p in ranked_passages if p[2] >= threshold]
        return filtered
    
    def filter_by_medical_keyword(self, ranked_passages: List[Tuple[str, Dict, float]], 
                                 user_query: str) -> List[Tuple[str, Dict, float]]:
        """
        Filter passages to ensure they contain medical keywords from the query.
        This helps ensure relevance for medical queries.
        """
        if not ranked_passages:
            return ranked_passages
            
        # Extract keywords from query (simple tokenization)
        query_words = set(user_query.lower().split())
        
        # Common medical terms to boost
        medical_terms = {
            'hypertension', 'blood', 'pressure', 'heart', 'cardiac', 'stroke', 
            'kidney', 'renal', 'diabetes', 'cancer', 'treatment', 'symptoms',
            'diagnosis', 'therapy', 'medication', 'drug', 'clinical', 'trial',
            'evidence', 'study', 'research', 'patient', 'disease', 'condition'
        }
        
        # Combine query words with medical terms
        relevant_terms = query_words.union(medical_terms)
        
        filtered = []
        for passage, meta, score in ranked_passages:
            passage_lower = passage.lower()
            # Check if passage contains at least one relevant term
            if any(term in passage_lower for term in relevant_terms):
                filtered.append((passage, meta, score))
        
        # If no passages match, return original (don't filter everything out)
        return filtered if filtered else ranked_passages
    
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
            source = "PubMed" if metadata.get("source_type") == "pubmed" else "Local"
            year = f" {metadata['year']}" if metadata.get("year") else ""
            context_parts.append(f"[Evidence {i}] ({source}{year}, confidence: {score:.3f})\n{passage}")

        return "\n\n".join(context_parts)
    
    # gate 1: query quality check
    def gate_1_query_quality(self, user_query: str, threshold: float = None) -> Dict[str, Any]:
        if threshold is None:
            threshold = config.THRESHOLDS["gate1"]
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
    
    # gate 2: retrieval sufficiency check 
    def gate_2_retrieval_sufficiency(self, query: str, retrieved_chunks: List[Dict[str, Any]], threshold: float = None) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
        if threshold is None:
            threshold = config.THRESHOLDS["gate2"]
        logger.info("Gate 2: Checking retrieval sufficiency...")
        
        # Get top passages by similarity (no energy/confidence filtering here)
        strong = sorted(retrieved_chunks, key=lambda x: x["similarity"], reverse=True)
        
        # Apply similarity threshold to get strong evidence
        threshold = max(0.4, config.THRESHOLDS["sim_threshold"] - 0.1)  # Dynamic threshold
        strong = [c for c in strong if c["similarity"] >= threshold][:10]  # Top 10 max
        
        # Guarantee minimum evidence (fallback to top-3 if threshold too strict)
        if len(strong) < 3:
            strong = sorted(retrieved_chunks, key=lambda x: x["similarity"], reverse=True)[:3]
        
        # BINARY CHECKS: Count and Coverage only
        count_ok = len(strong) >= config.THRESHOLDS["min_strong_evidence"]
        covered_concepts = self._get_covered_concepts(strong)
        coverage = len(covered_concepts) / 3.0  # disease, organ, mechanism
        coverage_ok = coverage >= config.THRESHOLDS["min_coverage"]
        
        # PURELY BINARY: Pass if count is sufficient OR if we have lots of evidence (bypass coverage)
        # This allows queries with many strong evidence pieces to pass even if concept extraction is incomplete
        passed = count_ok and (coverage_ok or len(strong) >= 5)
        
        gate_info = {
            "gate": "retrieval_sufficiency",
            "passed": passed,
            "energy_score": 1.0 if passed else 0.0,  # Binary: 1.0 for sufficient, 0.0 for insufficient
            "threshold": threshold,
            "num_strong_evidence": len(strong),
            "count_ok": count_ok,
            "coverage_ok": coverage_ok,
            "coverage": round(coverage, 2),  # Keep for error messages
            "covered_concepts": list(self._get_covered_concepts(strong)),
            "thresholds": {
                "similarity": config.THRESHOLDS["sim_threshold"],
                "min_strong_evidence": config.THRESHOLDS["min_strong_evidence"],
                "min_coverage": config.THRESHOLDS["min_coverage"]
            }
        }
        
        if passed:
            logger.info(f"Gate 2 PASSED: {len(strong)} strong evidence, sufficient for answer")
        else:
            logger.warning(f"Gate 2 FAILED: {len(strong)} strong evidence, insufficient for answer")
        
        return passed, gate_info, strong
    
    def _check_concept_coverage(self, chunks: List[Dict[str, Any]]) -> bool:
        """Check if chunks cover required medical concepts."""
        covered_concepts = self._get_covered_concepts(chunks)
        coverage = len(covered_concepts) / 3.0  # disease, organ, mechanism
        return coverage >= config.THRESHOLDS["min_coverage"]
    
    def _get_covered_concepts(self, chunks: List[Dict[str, Any]]) -> set:
        """Get set of covered medical concepts."""
        covered_concepts = set()
        for c in chunks:
            covered_concepts |= extract_concepts(c["text"])
        return covered_concepts
    
    # gate 3: evidence consistency check
    def gate_3_evidence_consistency(self, filtered_passages: List[Tuple[str, Dict, float]], 
                                   threshold: float = None) -> Dict[str, Any]:
        if threshold is None:
            threshold = config.THRESHOLDS["gate3"]
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
    
    def compute_confidence(self, strong_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not strong_evidence:
            return {
                "score": 0.0,
                "label": "LOW",
                "components": {
                    "mean_similarity": 0.0,
                    "mean_energy": 1.0,
                    "evidence_count": 0,
                    "coverage": 0.0,
                }
            }
        
        similarities = [c["similarity"] for c in strong_evidence]
        energies = [c["energy"] for c in strong_evidence]
        
        mean_sim = sum(similarities) / len(similarities)
        mean_energy = sum(energies) / len(energies)
        
        # coverage again
        covered_concepts = set()
        for c in strong_evidence:
            covered_concepts |= extract_concepts(c["text"])
        
        coverage = len(covered_concepts) / 3.0
        n = len(strong_evidence)
        
        # confidence formula (using normalized energy scores where lower = better)
        score = (
            0.35 * mean_sim +
            0.35 * (1.0 - mean_energy) +  # mean_energy is now normalized [0,1], lower = better
            0.20 * min(1.0, n / 3.0) +
            0.10 * coverage
        )
        
        if score >= 0.75:
            label = "HIGH"
        elif score >= 0.55:
            label = "MEDIUM"
        else:
            label = "LOW"
        
        return {
            "score": round(score, 3),
            "label": label,
            "components": {
                "mean_similarity": round(mean_sim, 3),
                "mean_energy": round(mean_energy, 3),
                "evidence_count": n,
                "coverage": round(coverage, 2),
            }
        }
    
    # verify each evidence with answer 
    def verify_evidence_chain(self, answer: str, filtered_passages: List[Tuple[str, Dict, float]],
                             min_entailment_score: float = None) -> Dict[str, Any]:
        if min_entailment_score is None:
            min_entailment_score = THRESHOLDS["min_entailment_score"]
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
        
        # Get passage texts
        passage_texts = [passage for passage, _, _ in filtered_passages]
        
        verification_results = []
        verified_count = 0
        
        for sentence_idx, sentence in enumerate(sentences, 1):
            # Check entailment with all passages using NLI
            entailments = []
            for i, passage in enumerate(passage_texts):
                # Use NLI to check if passage entails sentence
                result = self.nli_pipeline(f"{passage} </s> {sentence}")
                label = result[0]['label']
                score = result[0]['score']
                entailments.append((i, label, score))
            
            # Sort by entailment score (only consider ENTAILMENT)
            entailments = [(i, score) for i, label, score in entailments if label == "ENTAILMENT"]
            entailments.sort(key=lambda x: x[1], reverse=True)
            
            # Find best matching passage(s)
            supporting_passages = []
            for passage_idx, entail_score in entailments[:3]:  # Top 3 entailments
                if entail_score >= min_entailment_score:
                    supporting_passages.append({
                        "passage_number": passage_idx + 1,
                        "entailment_score": round(entail_score, 3),
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
                "confidence": supporting_passages[0]["entailment_score"] if supporting_passages else 0.0,
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
            "min_entailment_threshold": min_entailment_score
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
        output.append(f"   Threshold: {verification_data['min_entailment_threshold']}\n")
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
            
            # Transform conflicts to add passage indices
            if contradiction_data.get("conflicts"):
                for conflict in contradiction_data["conflicts"]:
                    # Extract passage index from source_id (e.g., "Source 1" -> 0, "Source 2" -> 1)
                    if "source_1" in conflict and "source_id" in conflict["source_1"]:
                        source_id = conflict["source_1"]["source_id"]
                        idx_match = re.search(r'Source (\d+)', source_id)
                        conflict["passage1_idx"] = int(idx_match.group(1)) - 1 if idx_match else 0
                    else:
                        conflict["passage1_idx"] = 0
                        
                    if "source_2" in conflict and "source_id" in conflict["source_2"]:
                        source_id = conflict["source_2"]["source_id"]
                        idx_match = re.search(r'Source (\d+)', source_id)
                        conflict["passage2_idx"] = int(idx_match.group(1)) - 1 if idx_match else 1
                    else:
                        conflict["passage2_idx"] = 1
                    
                    # Build explanation from the conflict data
                    source1_claim = conflict.get("source_1", {}).get("claim", "N/A")
                    source2_claim = conflict.get("source_2", {}).get("claim", "N/A")
                    topic = conflict.get("topic", "Unknown")
                    
                    conflict["explanation"] = f"{topic}: '{source1_claim}' vs '{source2_claim}'"
            
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
    
    def answer_query(self, user_query: str, top_k: int = None, threshold: float = None, 
                     use_llm: bool = True, detect_conflicts: bool = True,
                     enable_gates: bool = True,
                     gate_thresholds: Dict[str, float] = None,
                     verify_chain: bool = False) -> Dict[str, Any]:
        if top_k is None:
            top_k = THRESHOLDS["default_top_k"]
        if threshold is None:
            threshold = THRESHOLDS["similarity_threshold"]
        
        logger.info("="*80)
        logger.info(f"Query: {user_query}")
        logger.info("="*80)
        
        # Set default gate thresholds
        if gate_thresholds is None:
            gate_thresholds = {
                "gate1": THRESHOLDS["gate1"],   # Query quality
                "gate2": THRESHOLDS["gate2"],   # Retrieval quality  
                "gate3": THRESHOLDS["gate3"]    # Evidence consistency
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
        
        # Step 2: Rerank
        ranked = self.rerank_passages(user_query, passages, metadatas)
        logger.info(f"[2/5] Ranked {len(ranked)} passages")
        
        # Step 3: Compute energy scores
        ranked_passages = [p[0] for p in ranked]
        energies = self.compute_energy_scores(ranked_passages)
        logger.info(f"[3/5] Computed energy scores for {len(energies)} passages")
        
        # Create retrieved_chunks for gate 2
        retrieved_chunks = []
        for i, ((passage, metadata, similarity), energy) in enumerate(zip(ranked, energies)):
            retrieved_chunks.append({
                "text": passage,
                "similarity": similarity,
                "energy": energy,
                "metadata": metadata,
                "rank": i + 1
            })
            logger.debug(f"Chunk {i+1}: sim={similarity:.3f}, energy={energy:.3f}")
        
        # Initialize strong_evidence (either from gates or all chunks)
        strong_evidence = retrieved_chunks  # Default to all chunks when gates disabled
        
        if enable_gates:
            # Gate 2: Retrieval Sufficiency
            gate2_passed, gate2_info, strong_evidence = self.gate_2_retrieval_sufficiency(user_query, retrieved_chunks, gate_thresholds["gate2"])
            gate_results["gate2"] = gate2_info
            
            if not gate2_passed:
                logger.warning(" Gate 2 FAILED - Attempting PubMed fallback")
                try:
                    pubmed_ids = pubmed_search(user_query, max_results=5, years=2)
                    if pubmed_ids:
                        pubmed_abstracts = pubmed_fetch_abstracts(pubmed_ids)
                        logger.info(f"PubMed fallback: Retrieved {len(pubmed_abstracts)} abstracts")
                        
                        for abstract_data in pubmed_abstracts:
                            chunks = self._chunk_text(abstract_data["abstract"], chunk_size=200)
                            for chunk in chunks:
                                passages.append(chunk)
                                metadatas.append({
                                    "source_type": "pubmed",
                                    "pmid": abstract_data["pmid"],
                                    "title": abstract_data["title"],
                                    "year": abstract_data["year"],
                                    "topic_name": "PubMed"
                                })
                        
                        # Re-rerank with combined passages
                        ranked = self.rerank_passages(user_query, passages, metadatas)
                        energies = self.compute_energy_scores([p[0] for p in ranked])
                        
                        # Update retrieved_chunks
                        retrieved_chunks = []
                        for i, ((passage, metadata, similarity), energy) in enumerate(zip(ranked, energies)):
                            retrieved_chunks.append({
                                "text": passage,
                                "similarity": similarity,
                                "energy": energy,
                                "metadata": metadata,
                                "rank": i + 1
                            })
                        
                        # Re-run gate2
                        gate2_passed, gate2_info, strong_evidence = self.gate_2_retrieval_sufficiency(user_query, retrieved_chunks, gate_thresholds["gate2"])
                        gate_results["gate2"] = gate2_info
                        
                        if not gate2_passed:
                            logger.warning(" Gate 2 STILL FAILED after PubMed - Stopping pipeline")
                            diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate2")
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
                                "rejection_reason": f"Retrieval sufficiency too low even after PubMed fallback: {gate2_info['num_strong_evidence']} strong evidence, {gate2_info['coverage']:.1%} coverage",
                                "suggestion": "Insufficient evidence found. This may require specialized medical consultation.",
                                "compute_saved": "Saved: reranking + LLM",
                                "confidence": "rejected",
                                "diagnosis": diagnosis,
                                "gate_diagnostics": gate2_info
                            }
                    else:
                        logger.warning("No PubMed results - rejecting")
                        diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate2")
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
                            "rejection_reason": f"Retrieval sufficiency too low: {gate2_info['num_strong_evidence']} strong evidence, {gate2_info['coverage']:.1%} coverage",
                            "suggestion": "Insufficient evidence found. This may require specialized medical consultation.",
                            "compute_saved": "Saved: reranking + LLM",
                            "confidence": "rejected",
                            "diagnosis": diagnosis,
                            "gate_diagnostics": gate2_info
                        }
                except Exception as e:
                    logger.warning(f"PubMed fallback failed: {e} - rejecting")
                    diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate2")
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
                        "rejection_reason": f"Retrieval sufficiency too low: {gate2_info['num_strong_evidence']} strong evidence, {gate2_info['coverage']:.1%} coverage",
                        "suggestion": "Insufficient evidence found. This may require specialized medical consultation.",
                        "compute_saved": "Saved: reranking + LLM",
                        "confidence": "rejected",
                        "diagnosis": diagnosis,
                        "gate_diagnostics": gate2_info
                    }
        
        # Compute confidence from strong evidence
        confidence = self.compute_confidence(strong_evidence)
        logger.info(f"[4/6] Confidence: {confidence['label']} ({confidence['score']:.3f})")
        
        # Step 4: Filter (using strong evidence as filtered)
        filtered = [(c["text"], c["metadata"], c["similarity"]) for c in strong_evidence]
        logger.info(f"[5/6] Using {len(filtered)} strong evidence passages")
        
        if enable_gates:
            gate3_result = self.gate_3_evidence_consistency(filtered, gate_thresholds["gate3"])
            gate_results["gate3"] = gate3_result
            
            if not gate3_result["passed"]:
                logger.warning(" Gate 3 FAILED - Attempting PubMed fallback")
                # PubMed fallback already done in gate2, so just reject
                logger.warning(" Gate 3 STILL FAILED - Stopping pipeline")
                diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate3")
                context = self.prepare_for_llm(filtered) if filtered else "No evidence passed threshold"
                partial_evidence = filtered[:3] if filtered else []
                return {
                    "query": user_query,
                    "gate_status": "rejected_at_gate3",
                    "gates": gate_results,
                    "retrieved_count": len(passages),
                    "ranked_passages": ranked,
                    "filtered_passages": filtered,
                    "strong_evidence": strong_evidence,
                    "partial_evidence": partial_evidence,
                    "context": context,
                    "answer": None,
                    "rejection_reason": f"Evidence consistency too low: {', '.join(gate3_result['reasons'])}",
                    "suggestion": "Evidence is too weak or inconsistent to provide a confident answer. This requires personalized medical advice.",
                    "compute_saved": "Saved: LLM call",
                    "confidence": "rejected",
                    "diagnosis": diagnosis,
                    "gate_diagnostics": {
                        "gate": "evidence_consistency",
                        "passed": False,
                        "consistency_score": gate3_result.get("energy_score", 0.0),
                        "threshold": gate_thresholds["gate3"],
                        "reasons": gate3_result.get("reasons", []),
                        "num_strong_evidence": len(strong_evidence),
                        "coverage": gate2_info.get("coverage", 0.0) if 'gate2_info' in locals() else 0.0
                    },
                    "evidence_metadata": {
                        "num_sources": len(filtered),
                        "avg_similarity": gate3_result.get("metrics", {}).get("avg_similarity", 0.0),
                        "topics": list(set(m.get("topic_name", "") for _, m, _ in filtered if m.get("topic_name")))
                    }
                }
        
        # All gates passed - continue with normal pipeline
        if enable_gates:
            logger.info("All gates PASSED - Proceeding to LLM")
            
            # Calculate overall energy score
            overall_energy = (gate_results["gate1"]["energy_score"] + 
                            gate_results["gate2"]["energy_score"] + 
                            gate_results["gate3"]["energy_score"]) / 3
            logger.info(f"Overall energy score: {overall_energy:.2f}")
        else:
            overall_energy = 1.0  # Gates disabled
        
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
            "overall_energy": overall_energy if enable_gates else None,
            "retrieved_count": len(passages),
            "ranked_passages": ranked,
            "filtered_passages": filtered,
            "strong_evidence": strong_evidence,
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
            logger.info(f"[6/6] Querying LLM...")
            start_time = datetime.now()
            llm_result = self.query_llm(user_query, context)
            if llm_result is None:
                logger.error("LLM query returned None")
                answer, follow_up_questions = None, []
            else:
                answer, follow_up_questions = llm_result
            end_time = datetime.now()
            result["answer"] = answer
            result["follow_up_questions"] = follow_up_questions
            result["answer_time"] = (end_time - start_time).total_seconds()
            logger.info(f"[Done] Answer generated in {result['answer_time']:.2f}s with {len(follow_up_questions)} follow-up questions")
            
            # Step 6: Verify evidence chain (if requested and answer exists)
            if verify_chain and answer:
                logger.info("[6/6] Verifying evidence chain...")
                evidence_chain = self.verify_evidence_chain(answer, filtered, min_entailment_score=0.7)
                result["evidence_chain"] = evidence_chain
                
                # Warn if verification rate is low
                if evidence_chain["verification_rate"] < 0.8:
                    logger.warning(
                        f" LOW VERIFICATION RATE: {evidence_chain['verification_rate']*100:.1f}% - "
                        f"{evidence_chain['unverified_sentences']} sentence(s) may be hallucinated"
                    )
        else:
            logger.info(f"[5/6] Skipping LLM (use_llm={use_llm}, filtered={len(filtered)>0})")
        
        logger.info(f"About to return result with keys: {list(result.keys()) if result else 'None'}")
        return result
    
    def query_llm(self, user_query: str, context: str, model: str = "openai/gpt-4o-mini", 
                  temperature: float = None) -> tuple:
        if temperature is None:
            temperature = THRESHOLDS["llm_temperature"]
        if self.enable_cache:
            cache_key = self._get_cache_key("llm", user_query, context, model, temperature)
            if cache_key in self.cache:
                self.cache_stats["llm_hits"] += 1
                cached_result = self.cache[cache_key]
                if cached_result is not None:
                    # Handle old cached format (just answer string) vs new format (tuple)
                    if isinstance(cached_result, tuple):
                        return cached_result
                    else:
                        # Old cache format - return with empty follow-up questions
                        return cached_result, []
                else:
                    # Cached None - treat as miss
                    self.cache_stats["llm_misses"] += 1
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
**I can also answer:**
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
        
        if answer is None:
            logger.error("LLM returned None content")
            return None
        
        # Extract follow-up questions if present
        follow_up_questions = []
        if answer and ("I can also answer:" in answer or "Suggested Follow-up Questions:" in answer or "Follow-up Questions:" in answer):
            import re
            # Split on the follow-up questions marker (support both old and new formats)
            parts = re.split(r'\*\*I can also answer:\*\*|\*\*Suggested Follow-up Questions:\*\*|Follow-up Questions:|I can also answer:', answer)
            if len(parts) > 1:
                questions_section = parts[-1]
                # Extract numbered questions
                question_matches = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', questions_section, re.DOTALL)
                follow_up_questions = [q.strip().strip('[]').strip() for q in question_matches if q.strip()]
        
        result = (answer, follow_up_questions)
        
        if self.enable_cache:
            self.cache.set(cache_key, result, expire=self.cache_ttl)
        
        return result
    
    def generate_differential_diagnosis(self, user_query: str, top_k: int = None, 
                                       threshold: float = None, num_diagnoses: int = None) -> Dict[str, Any]:
        if top_k is None:
            top_k = THRESHOLDS["default_top_k"]
        if threshold is None:
            threshold = THRESHOLDS["similarity_threshold"]
        if num_diagnoses is None:
            num_diagnoses = THRESHOLDS["num_diagnoses"]
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
    parser.add_argument("--top_k", type=int, default=THRESHOLDS["default_top_k"], help="Top-k passages to retrieve")
    parser.add_argument("--threshold", type=float, default=THRESHOLDS["similarity_threshold"], help="Similarity threshold to keep passages")
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
        device=None,  # Will auto-detect CUDA
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