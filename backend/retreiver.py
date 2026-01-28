import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import logging
logger = logging.getLogger("retriever")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

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
from pathlib import Path
import requests
import numpy as np

from config import (
    MODEL_NAME, CHROMA_PATH, CHROMA_COLLECTION_NAME, RANKING_SCORER_CKPT, VERIFICATION_SCORER_CKPT, CACHE_DIR, THRESHOLDS
)

def pubmed_search(query: str, max_results: int = 5, years: int = 2) -> List[str]:
    try:
        import time
        from datetime import datetime

        # Calculate date range
        current_year = datetime.now().year
        start_year = current_year - years

        # Build PubMed query
        pubmed_query = f"({query}) AND ({start_year}[Date - Publication]:{current_year}[Date - Publication])"

        # PubMed E-utilities base URL
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi"

        params = {
            "db": "pubmed",
            "term": pubmed_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }

        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])

        logger.info(f"PubMed search for '{query}' returned {len(pmids)} results")
        return pmids

    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []

def pubmed_fetch_abstracts(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    try:
        # PubMed E-utilities base URL
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        fetch_url = f"{base_url}efetch.fcgi"

        # Fetch in batches of 10 to avoid API limits
        all_abstracts = []
        batch_size = 10

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            pmids_str = ",".join(batch_pmids)

            params = {
                "db": "pubmed",
                "id": pmids_str,
                "retmode": "xml",
                "rettype": "abstract"
            }

            response = requests.get(fetch_url, params=params, timeout=15)
            response.raise_for_status()

            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else "Unknown"

                    # Extract title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "Unknown Title"

                    # Extract publication year
                    year_elem = article.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None else "Unknown"

                    # Extract abstract
                    abstract_elem = article.find(".//AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else ""

                    if abstract and len(abstract.strip()) > 50:  # Only include substantial abstracts
                        all_abstracts.append({
                            "pmid": pmid,
                            "title": title,
                            "year": year,
                            "abstract": abstract.strip()
                        })

                except Exception as e:
                    logger.warning(f"Failed to parse PubMed article: {e}")
                    continue

            # Rate limiting
            import time
            time.sleep(0.5)

        logger.info(f"Successfully fetched {len(all_abstracts)} PubMed abstracts")
        return all_abstracts

    except Exception as e:
        logger.error(f"PubMed abstract fetch failed: {e}")
        return []

def extract_concepts(text: str) -> set:
    """Extract medical concepts from text (disease, organ, mechanism)."""
    concepts = set()
    text_lower = text.lower()
    
    # Disease-related keywords
    disease_keywords = ['disease', 'condition', 'disorder', 'syndrome', 'illness', 'pathology', 'diagnosis']
    if any(kw in text_lower for kw in disease_keywords):
        concepts.add('disease')
    
    # Organ/system keywords
    organ_keywords = ['heart', 'kidney', 'liver', 'lung', 'brain', 'organ', 'tissue', 'system', 'cardiac', 'renal', 'hepatic', 'pulmonary', 'neurological']
    if any(kw in text_lower for kw in organ_keywords):
        concepts.add('organ')
    
    # Mechanism/pathway keywords
    mechanism_keywords = ['mechanism', 'pathway', 'process', 'cause', 'effect', 'treatment', 'therapy', 'intervention', 'management', 'prevention']
    if any(kw in text_lower for kw in mechanism_keywords):
        concepts.add('mechanism')
    
    return concepts

# Setup logger
logger = logging.getLogger("retriever")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

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
        
        # Load separate encoders for ranking and verification to avoid device issues
        ranking_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        verification_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Load ranking scorer
        self.ranking_model = RankingScorer(ranking_encoder, embedding_dim=config.THRESHOLDS["embedding_dim"]).to(self.device)
        # Use weights_only=True to avoid loading arbitrary pickled objects
        self.ranking_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.ranking_model.eval()
        logger.info(f"Loaded ranking model from {checkpoint_path}")
        
        # Load verification scorer (separate model for verification)
        self.verification_model = RankingScorer(verification_encoder, embedding_dim=config.THRESHOLDS["embedding_dim"]).to(self.device)
        self.verification_model.load_state_dict(torch.load(verification_checkpoint_path, map_location=self.device, weights_only=True))
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
        
    def prepare_for_llm(self, filtered_passages: List[Tuple[str, Dict, float]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Prepare context for LLM with structured sources.
        
        Returns:
            Tuple of (context_string, sources_list)
        """
        if not filtered_passages:
            return "No relevant evidence found.", []

        seen = set()
        deduped = []
        sources = []
        
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
            # Add to context with citation number
            context_parts.append(f"[{i}] {passage}")
            
            # Build source entry for LangChain toolcall
            source_entry = {
                "citation_number": i,
                "title": metadata.get("title", f"Source {i}"),
                "source_type": metadata.get("source_type", "Unknown"),
                "year": metadata.get("year", "Unknown"),
                "authors": metadata.get("authors", "Unknown"),
                "url": metadata.get("url", ""),
                "pmid": metadata.get("pmid", ""),
                "confidence_score": round(score, 3),
                "topic": metadata.get("topic_name", "General")
            }
            sources.append(source_entry)

        context = "\n\n".join(context_parts)
        return context, sources
    
    def energy_aware_filtering(self, ranked_passages, energies, query):
        # Combine similarity + energy into unified score
        combined_scores = []
        for (passage, metadata, similarity), energy in zip(ranked_passages, energies):
            # Energy-weighted score: balance relevance (similarity) with quality (1-energy)
            # Lower energy = higher quality, so we use (1-energy)
            alpha = 0.6  # Weight for similarity
            beta = 0.4   # Weight for energy quality
            unified_score = alpha * similarity + beta * (1 - energy)
            
            combined_scores.append({
                'passage': passage,
                'metadata': metadata,
                'similarity': similarity,
                'energy': energy,
                'unified_score': unified_score,
                'quality_tier': self._classify_energy_tier(energy)
            })
        
        # Sort by unified score (best first)
        combined_scores.sort(key=lambda x: x['unified_score'], reverse=True)
        
        # ENERGY-BASED ADAPTIVE THRESHOLDING
        # Find the "energy gap" - where quality drops significantly
        energy_threshold = self._find_energy_gap(combined_scores)
        
        # Filter: Keep passages above energy threshold
        high_quality = [
            p for p in combined_scores 
            if p['energy'] <= energy_threshold  # Lower energy = better
        ]
        
        # Guarantee minimum evidence (take top 3 even if energy is high)
        if len(high_quality) < 3:
            high_quality = combined_scores[:3]
        
        logger.info(f"Energy filtering: {len(high_quality)}/{len(combined_scores)} passages "
                    f"passed threshold {energy_threshold:.3f}")
        
        return high_quality, energy_threshold

    def _classify_energy_tier(self, energy: float) -> str:
        # Remember: energy is normalized [0,1], lower = better
        if energy <= 0.3:
            return "GOLD"      # High-quality clinical evidence
        elif energy <= 0.5:
            return "SILVER"    # Good quality, usable
        elif energy <= 0.7:
            return "BRONZE"    # Moderate quality
        else:
            return "WEAK"      # Low quality, use with caution

    def _find_energy_gap(self, scored_passages, min_gap=0.15):
        if len(scored_passages) < 2:
            return 0.5  # Default
        
        energies = [p['energy'] for p in scored_passages]
        
        # Find largest consecutive gap
        max_gap = 0
        gap_index = 0
        
        for i in range(len(energies) - 1):
            gap = abs(energies[i+1] - energies[i])
            if gap > max_gap:
                max_gap = gap
                gap_index = i
        
        # If no significant gap found, use median
        if max_gap < min_gap:
            return sorted(energies)[len(energies)//2]
        
        # Threshold = energy level after the gap
        threshold = energies[gap_index]
        logger.info(f"Energy gap detected: {max_gap:.3f} at index {gap_index}")
        
        return threshold
          
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
            
    def energy_gate_2_retrieval_quality(self, passages_with_energy, query):
        if not passages_with_energy:
            return 0.0, {"reason": "No passages retrieved"}
        
        energies = [p['energy'] for p in passages_with_energy]
        
        # ENERGY LANDSCAPE METRICS
        
        # 1. Energy minimum (best evidence quality)
        min_energy = min(energies)
        
        # 2. Energy concentration (how many good passages?)
        gold_tier_count = sum(1 for e in energies if e <= 0.3)
        silver_tier_count = sum(1 for e in energies if 0.3 < e <= 0.5)
        
        # 3. Energy spread (consistency)
        energy_std = np.std(energies) if len(energies) > 1 else 0
        
        # 4. Energy depth (top-k quality)
        top5_avg_energy = np.mean(energies[:5])
        
        # COMPUTE CONFIDENCE SCORE
        # Each component contributes to overall confidence
        
        # Component 1: Best evidence quality (40%)
        quality_score = 1.0 - min_energy  # Lower energy = higher score
        
        # Component 2: Evidence breadth (30%)
        breadth_score = min(1.0, (gold_tier_count + silver_tier_count) / 5.0)
        
        # Component 3: Consistency (20%)
        consistency_score = 1.0 - min(1.0, energy_std / 0.3)  # Penalize high variance
        
        # Component 4: Top-k quality (10%)
        topk_score = 1.0 - top5_avg_energy
        
        confidence = (
            0.40 * quality_score +
            0.30 * breadth_score +
            0.20 * consistency_score +
            0.10 * topk_score
        )
        
        # Diagnostic info
        gate_info = {
            "confidence": round(confidence, 3),
            "passed": confidence >= 0.5,  # Soft threshold
            "energy_min": round(min_energy, 3),
            "gold_tier_count": gold_tier_count,
            "silver_tier_count": silver_tier_count,
            "energy_std": round(energy_std, 3),
            "top5_avg_energy": round(top5_avg_energy, 3),
            "components": {
                "quality": round(quality_score, 3),
                "breadth": round(breadth_score, 3),
                "consistency": round(consistency_score, 3),
                "topk": round(topk_score, 3)
            },
            "interpretation": self._interpret_energy_landscape(
                min_energy, gold_tier_count, energy_std
            )
        }
        
        logger.info(f"🎯 Gate 2 Energy Confidence: {confidence:.3f}")
        logger.info(f"   Energy landscape: min={min_energy:.3f}, "
                    f"gold={gold_tier_count}, spread={energy_std:.3f}")
        
        return confidence, gate_info

    def _interpret_energy_landscape(self, min_energy, gold_count, std):
        interpretations = []
        
        if min_energy <= 0.2:
            interpretations.append("✅ Excellent evidence quality detected")
        elif min_energy <= 0.4:
            interpretations.append("✓ Good evidence quality")
        else:
            interpretations.append("⚠️ Evidence quality concerns")
        
        if gold_count >= 3:
            interpretations.append(f"✅ Strong evidence base ({gold_count} gold-tier sources)")
        elif gold_count >= 1:
            interpretations.append(f"✓ Moderate evidence ({gold_count} gold-tier)")
        else:
            interpretations.append("⚠️ No gold-tier evidence found")
        
        if std <= 0.15:
            interpretations.append("✅ Consistent evidence quality")
        elif std <= 0.25:
            interpretations.append("✓ Acceptable quality variance")
        else:
            interpretations.append("⚠️ High quality variance - mixed evidence")
        
        return interpretations

    def energy_gate_3_coherence(self, passages_with_energy):
        if len(passages_with_energy) < 2:
            return 0.5, {"reason": "Insufficient passages for coherence check"}
        
        energies = [p['energy'] for p in passages_with_energy]
        
        # Pairwise energy similarity
        coherence_scores = []
        for i in range(len(energies) - 1):
            for j in range(i + 1, min(i + 4, len(energies))):  # Check neighbors
                energy_diff = abs(energies[i] - energies[j])
                coherence = 1.0 - min(1.0, energy_diff / 0.5)  # Normalize
                coherence_scores.append(coherence)
        
        avg_coherence = np.mean(coherence_scores)
        
        # Energy-based consensus detection
        # If most passages cluster at similar energy levels, coherence is high
        energy_clusters = self._detect_energy_clusters(energies)
        largest_cluster_pct = max(len(c) for c in energy_clusters) / len(energies)
        
        # Final coherence score
        confidence = 0.6 * avg_coherence + 0.4 * largest_cluster_pct
        
        gate_info = {
            "confidence": round(confidence, 3),
            "passed": confidence >= 0.5,
            "avg_pairwise_coherence": round(avg_coherence, 3),
            "largest_cluster_pct": round(largest_cluster_pct, 3),
            "num_clusters": len(energy_clusters),
            "interpretation": (
                "Strong consensus" if largest_cluster_pct >= 0.6 else
                "Moderate agreement" if largest_cluster_pct >= 0.4 else
                "Evidence fragmentation detected"
            )
        }
        
        logger.info(f"🎯 Gate 3 Energy Coherence: {confidence:.3f}")
        
        return confidence, gate_info

    def _detect_energy_clusters(self, energies, threshold=0.15):
        sorted_energies = sorted(energies)
        clusters = []
        current_cluster = [sorted_energies[0]]
        
        for energy in sorted_energies[1:]:
            if energy - current_cluster[-1] <= threshold:
                current_cluster.append(energy)
            else:
                clusters.append(current_cluster)
                current_cluster = [energy]
        
        clusters.append(current_cluster)
        return clusters
    
    def visualize_energy_landscape(self, passages_with_energy, query):
        if not passages_with_energy:
            return "No energy data to visualize"
        
        output = []
        output.append("ENERGY LANDSCAPE ANALYSIS")
        
        energies = [p['energy'] for p in passages_with_energy]
        
        # Energy distribution histogram (ASCII)
        output.append("Energy Distribution (lower = better quality):")
        output.append("")
        
        # Bin energies into tiers
        bins = {
            'GOLD (0.0-0.3)': sum(1 for e in energies if e <= 0.3),
            'SILVER (0.3-0.5)': sum(1 for e in energies if 0.3 < e <= 0.5),
            'BRONZE (0.5-0.7)': sum(1 for e in energies if 0.5 < e <= 0.7),
            'WEAK (0.7-1.0)': sum(1 for e in energies if e > 0.7)
        }
        
        max_count = max(bins.values()) if bins.values() else 1
        
        for tier, count in bins.items():
            bar_length = int((count / max_count) * 40) if max_count > 0 else 0
            bar = '█' * bar_length
            
            # Color coding (using emojis since we're in terminal)
            icon = {
                'GOLD (0.0-0.3)': '🟢',
                'SILVER (0.3-0.5)': '🔵',
                'BRONZE (0.5-0.7)': '🟡',
                'WEAK (0.7-1.0)': '🔴'
            }.get(tier, '⚪')
            
            output.append(f"{icon} {tier:20s} [{count:2d}] {bar}")
        
        output.append("")
        
        # Key statistics
        output.append("📊 Energy Statistics:")
        output.append(f"   Best (min):     {min(energies):.3f}")
        output.append(f"   Average:        {np.mean(energies):.3f}")
        output.append(f"   Worst (max):    {max(energies):.3f}")
        output.append(f"   Std Dev:        {np.std(energies):.3f}")
        output.append(f"   Total passages: {len(energies)}")
        output.append("")
        
        # Top passages ranked by energy
        output.append("Top 5 Highest Quality (by energy):")
        sorted_passages = sorted(passages_with_energy, key=lambda x: x['energy'])
        
        for i, p in enumerate(sorted_passages[:5], 1):
            energy = p['energy']
            tier = p['quality_tier']
            similarity = p.get('similarity', 0)
            
            # Show passage preview
            passage_preview = p['passage'][:80].replace('\n', ' ')
            
            tier_icon = {
                'GOLD': 'G',
                'SILVER': 'S',
                'BRONZE': 'B',
                'WEAK': 'W'
            }.get(tier, 'U')
            
            output.append(f"\n{i}. {tier_icon} {tier} | Energy: {energy:.3f} | Similarity: {similarity:.3f}")
            output.append(f"   \"{passage_preview}...\"")
        
        output.append("\n" + "="*80)
        
        # Interpretation
        avg_energy = np.mean(energies)
        gold_count = bins['GOLD (0.0-0.3)']
        
        if avg_energy <= 0.4 and gold_count >= 3:
            assessment = "EXCELLENT - High-quality evidence base"
            recommendation = "Proceed with high confidence"
        elif avg_energy <= 0.5 and gold_count >= 1:
            assessment = "GOOD - Sufficient quality for medical guidance"
            recommendation = "Proceed with moderate confidence"
        elif avg_energy <= 0.6:
            assessment = "MODERATE - Quality concerns present"
            recommendation = "Use with caution, verify key claims"
        else:
            assessment = "WEAK - Insufficient quality"
            recommendation = "Recommend seeking additional sources"
        
        output.append(f"\n Overall Assessment: {assessment}")
        output.append(f"   {recommendation}")
        output.append("")
        
        return "\n".join(output)

    def explain_energy_score(self, energy: float, tier: str) -> str:
        explanations = {
            'GOLD': (
                "This evidence has EXCELLENT quality characteristics:\n"
                "  • Strongly resembles peer-reviewed clinical evidence\n"
                "  • Dense medical terminology and structured reporting\n"
                "  • High confidence for medical decision support"
            ),
            'SILVER': (
                "This evidence has GOOD quality:\n"
                "  • Resembles clinical documentation or guidelines\n"
                "  • Acceptable for general medical information\n"
                "  • Moderate confidence for guidance"
            ),
            'BRONZE': (
                "This evidence has MODERATE quality:\n"
                "  • May be educational or general health content\n"
                "  • Less formal structure than clinical evidence\n"
                "  • Use as supporting information only"
            ),
            'WEAK': (
                "This evidence has LOW quality:\n"
                "  • May be fragmentary or low-information content\n"
                "  • Does not strongly resemble clinical evidence\n"
                "  • Not recommended for medical decisions"
            )
        }
        
        return f"Energy: {energy:.3f} [{tier}]\n{explanations.get(tier, 'Quality unknown')}"
    
    def compute_confidence(self, strong_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not strong_evidence:
            return {
                "score": 0.0,
                "label": "VERY_LOW",
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
        
        # Check for high-quality evidence (from GRADE assessment)
        high_quality_count = 0
        for evidence in strong_evidence:
            metadata = evidence.get("metadata", {})
            grade = metadata.get("grade_level", "UNKNOWN")
            if grade in ["HIGH", "MODERATE"]:
                high_quality_count += 1
        
        # Quality bonus: if we have high/moderate quality evidence, boost confidence
        quality_bonus = min(0.25, high_quality_count * 0.15)
        
        n = len(strong_evidence)
        coverage = 0.5  # Default moderate coverage
        
        # Improved confidence formula:
        # - More weight on evidence count (consistency indicator)
        # - Less weight on energy (which can be noisy)
        # - Add quality bonus for high-quality sources
        score = (
            0.30 * mean_sim +              # Similarity to query
            0.25 * (1.0 - mean_energy) +   # Energy (lower is better)
            0.30 * min(1.0, n / 5.0) +     # Evidence count (5+ = max)
            0.15 * coverage +               # Coverage
            quality_bonus                   # Bonus for high-quality sources
        )
        
        # Confidence thresholds - more generous to reflect actual quality
        if score >= 0.70:
            label = "HIGH"
        elif score >= 0.50:
            label = "MEDIUM"
        elif score >= 0.30:
            label = "LOW"
        else:
            label = "VERY_LOW"
        
        return {
            "score": round(score, 3),
            "label": label,
            "components": {
                "mean_similarity": round(mean_sim, 3),
                "mean_energy": round(mean_energy, 3),
                "evidence_count": n,
                "high_quality_count": high_quality_count,
                "quality_bonus": round(quality_bonus, 3),
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
        
        if "gate2" in gate_status:
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
        
        # QUICK CHECK: If all passages are identical, no contradictions possible
        passages_text = [p[0].strip().lower() for p in filtered_passages[:10]]
        if len(set(passages_text)) == 1:
            logger.info("All passages are identical - no contradictions possible")
            return {
                "has_contradictions": False,
                "conflicts": [],
                "note": "All evidence sources are identical"
            }
        
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
        system_prompt = """You are a strict medical evidence analyzer. Your ONLY job is to detect REAL contradictions.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ONLY report contradictions that ACTUALLY APPEAR in the provided evidence
2. Do NOT fabricate, invent, or assume any claims not explicitly stated
3. Do NOT make up conflicts about topics not mentioned
4. Compare ONLY the exact text provided - no external knowledge
5. If sources say the same thing (even with minor wording differences), there is NO contradiction
6. Every conflict must quote EXACT TEXT from the evidence

WHAT COUNTS AS A REAL CONTRADICTION:
- Source A says "BP target <130/80" AND Source B says "BP target <140/90" (both topics in evidence)
- Source A says "avoid sodium" AND Source B says "sodium is safe" (both in evidence)
- Different dosages for the same drug (all in evidence)

WHAT IS NOT A CONTRADICTION:
- Similar information phrased differently
- Same evidence repeated
- Different detail levels (summary vs detailed)
- My own assumptions about what the sources might disagree on

RESPONSE FORMAT (JSON only):
{
  "has_contradictions": false,
  "conflicts": [],
  "note": "All sources say essentially the same thing" or "No contradictions found"
}

Or if real contradictions exist:
{
  "has_contradictions": true,
  "conflicts": [
    {
      "topic": "Exact topic from evidence",
      "source_1": {
        "claim": "Exact quote from Source 1",
        "source_id": "Source 1"
      },
      "source_2": {
        "claim": "Exact quote from Source 2",
        "source_id": "Source 2"
      },
      "severity": "critical or moderate",
      "explanation": "Why these contradict"
    }
  ]
}

REMEMBER: If you're not 100% certain a contradiction exists in the text, report has_contradictions: false"""
        
        user_prompt = f"""Query: {user_query}

EVIDENCE TO ANALYZE (analyze ONLY this text):
{evidence_text}

CRITICAL: Only report contradictions that appear in the evidence above.
Do not make up claims or invent contradictions.
Return valid JSON only."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0  # Deterministic - no creativity/hallucinations
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                contradiction_data = json.loads(json_match.group())
            else:
                contradiction_data = json.loads(response_text)
            
            # DEFENSIVE CHECK: Verify contradictions actually exist in evidence
            if contradiction_data.get("has_contradictions") and contradiction_data.get("conflicts"):
                validated_conflicts = []
                for conflict in contradiction_data["conflicts"]:
                    source1_claim = conflict.get("source_1", {}).get("claim", "").lower()
                    source2_claim = conflict.get("source_2", {}).get("claim", "").lower()
                    
                    # Check if both claims actually appear in the evidence
                    evidence_lower = evidence_text.lower()
                    claim1_in_evidence = source1_claim and source1_claim in evidence_lower
                    claim2_in_evidence = source2_claim and source2_claim in evidence_lower
                    
                    if claim1_in_evidence and claim2_in_evidence:
                        validated_conflicts.append(conflict)
                    else:
                        logger.warning(f"Filtered out hallucinated contradiction: {conflict.get('topic')}")
                
                # Update with only validated conflicts
                if not validated_conflicts:
                    logger.info("Contradiction detection: LLM reported conflicts but none validated")
                    contradiction_data["has_contradictions"] = False
                    contradiction_data["conflicts"] = []
                    contradiction_data["note"] = "Initial conflicts reported by LLM were not found in actual evidence"
                else:
                    contradiction_data["conflicts"] = validated_conflicts
            
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
                "gate2": THRESHOLDS["gate2"],   # Retrieval quality  
                "gate3": THRESHOLDS["gate3"]    # Evidence consistency
            }
        
        gate_results = {}
        
        # Skip Gate 1 (query quality check) - removed
        
        # Step 1: Retrieve
        passages, metadatas = self.retrieve_top_k(user_query, top_k)
        logger.info(f"[1/5] Retrieved {len(passages)} passages")
        
        # Step 2: Rerank
        ranked = self.rerank_passages(user_query, passages, metadatas)
        logger.info(f"[2/5] Ranked {len(ranked)} passages")
        
        # Step 3: Compute energy scores
        ranked_passages = [p[0] for p in ranked]
        energies = self.compute_energy_scores(ranked_passages)
        logger.info(f"[3/6] Computed energy scores for {len(energies)} passages")
        
        # Step 4: Energy-aware filtering
        high_quality, energy_threshold = self.energy_aware_filtering(ranked, energies, user_query)
        logger.info(f"[4/6] Energy filtering: {len(high_quality)} high-quality passages")
        
        # Visualize energy landscape for user understanding
        energy_visualization = self.visualize_energy_landscape(high_quality, user_query)
        
        # Create strong_evidence from energy-filtered results
        strong_evidence = []
        for i, item in enumerate(high_quality):
            strong_evidence.append({
                "text": item['passage'],
                "similarity": item['similarity'],
                "energy": item['energy'],
                "metadata": item['metadata'],
                "rank": i + 1,
                "unified_score": item['unified_score'],
                "quality_tier": item['quality_tier']
            })
        
        # Save original strong evidence for fallback
        original_strong_evidence = strong_evidence.copy()
        
        if enable_gates:
            # ENERGY-BASED GATE 2: Quality Assessment Only
            energy_confidence, energy_info = self.energy_gate_2_retrieval_quality(strong_evidence, user_query)
            gate2_passed = energy_confidence >= 0.5
            
            # Format energy gate result to match expected structure
            gate2_info = {
                "gate": "energy_quality",
                "passed": gate2_passed,
                "energy_score": energy_confidence,
                "threshold": 0.5,
                "num_strong_evidence": len(strong_evidence),
                "energy_confidence": energy_confidence,
                "energy_landscape": energy_info,
                "combined_score": energy_confidence,
                "decision_factors": {
                    "energy": round(energy_confidence, 3)
                }
            }
            gate_results["gate2"] = gate2_info
            
            if not gate2_passed:
                logger.warning(" Gate 2 FAILED - Attempting PubMed fallback")
                try:
                    pubmed_ids = pubmed_search(user_query, max_results=5, years=2)
                    pubmed_abstracts = pubmed_fetch_abstracts(pubmed_ids) if pubmed_ids else []
                    logger.info(f"PubMed fallback: Retrieved {len(pubmed_abstracts)} abstracts")
                    
                    # Always add PubMed data if available, even if empty
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
                    
                    # Re-rerank with combined passages (always do this if we attempted PubMed)
                    if pubmed_abstracts or pubmed_ids:  # If we got any PubMed response
                        ranked = self.rerank_passages(user_query, passages, metadatas)
                        energies = self.compute_energy_scores([p[0] for p in ranked])
                        
                        # Re-apply energy filtering
                        high_quality, energy_threshold = self.energy_aware_filtering(ranked, energies, user_query)
                        
                        # Update strong_evidence
                        strong_evidence = []
                        for i, item in enumerate(high_quality):
                            strong_evidence.append({
                                "text": item['passage'],
                                "similarity": item['similarity'],
                                "energy": item['energy'],
                                "metadata": item['metadata'],
                                "rank": i + 1,
                                "unified_score": item['unified_score'],
                                "quality_tier": item['quality_tier']
                            })
                        
                        # Re-run ENERGY GATE 2 on updated evidence
                        energy_confidence, energy_info = self.energy_gate_2_retrieval_quality(strong_evidence, user_query)
                        gate2_passed = energy_confidence >= 0.5
                        
                        gate2_info = {
                            "gate": "energy_quality",
                            "passed": gate2_passed,
                            "energy_score": energy_confidence,
                            "threshold": 0.5,
                            "num_strong_evidence": len(strong_evidence),
                            "energy_confidence": energy_confidence,
                            "energy_landscape": energy_info,
                            "combined_score": energy_confidence
                        }
                        gate_results["gate2"] = gate2_info
                        
                except Exception as e:
                    logger.error(f"PubMed fallback failed: {e}")
                    # Continue with original evidence if PubMed fails
                    strong_evidence = original_strong_evidence[:3]  # Take top 3 as fallback
        else:
            gate_results = None
        
        # Compute confidence from strong evidence
        confidence = self.compute_confidence(strong_evidence)
        logger.info(f"[5/6] Confidence: {confidence['label']} ({confidence['score']:.3f})")
        
        # Step 6: Filter (using strong evidence as filtered)
        filtered = [(c["text"], c["metadata"], c["similarity"]) for c in strong_evidence]
        logger.info(f"[6/6] Using {len(filtered)} strong evidence passages")
        
        if enable_gates:
            # ENERGY-BASED COHERENCE CHECK ONLY
            if len(strong_evidence) >= 2:
                energy_coherence, coherence_info = self.energy_gate_3_coherence(strong_evidence)
                
                # Format energy coherence result to match gate3 structure
                gate3_result = {
                    "gate": "energy_coherence",
                    "passed": coherence_info["passed"],
                    "energy_score": energy_coherence,
                    "threshold": 0.5,  # Energy coherence threshold
                    "reasons": [coherence_info["interpretation"]],
                    "action": "proceed" if coherence_info["passed"] else "reject",
                    "metrics": {
                        "num_passages": len(strong_evidence),
                        "avg_pairwise_coherence": coherence_info["avg_pairwise_coherence"],
                        "largest_cluster_pct": coherence_info["largest_cluster_pct"]
                    },
                    "energy_coherence": energy_coherence,
                    "coherence_info": coherence_info,
                    "combined_score": energy_coherence
                }
                gate_results["gate3"] = gate3_result
            else:
                # Not enough evidence for coherence check
                gate3_result = {
                    "gate": "energy_coherence",
                    "passed": False,
                    "energy_score": 0.0,
                    "threshold": 0.5,
                    "reasons": ["Insufficient passages for coherence analysis"],
                    "action": "reject",
                    "metrics": {"num_passages": len(strong_evidence)},
                    "combined_score": 0.0
                }
                gate_results["gate3"] = gate3_result
                logger.info(f"🎯 Gate 3: Insufficient passages ({len(strong_evidence)}) for coherence check")
            
            if not gate3_result["passed"]:
                logger.warning(" Gate 3 FAILED - Attempting PubMed fallback")
                # PubMed fallback already done in gate2, so just reject
                logger.warning(" Gate 3 STILL FAILED - Stopping pipeline")
                diagnosis = self._diagnose_low_energy(gate_results, "rejected_at_gate3")
                context_result = self.prepare_for_llm(filtered) if filtered else ("No evidence passed threshold", [])
                context = context_result[0] if isinstance(context_result, tuple) else context_result
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
            
            # Calculate overall energy score (gates 2 and 3 only)
            overall_energy = (gate_results["gate2"]["energy_score"] + 
                            gate_results["gate3"]["energy_score"]) / 2
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
        context_result = self.prepare_for_llm(filtered)
        if isinstance(context_result, tuple):
            context, sources_list = context_result
        else:
            context = context_result
            sources_list = []
        
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
            "sources": sources_list,
            "sources_display": self.format_sources_for_display(sources_list),
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
            },
            "energy_visualization": energy_visualization
        }
        
        if use_llm and filtered:
            logger.info(f"[6/6] Querying LLM...")
            start_time = datetime.now()
            llm_result, sources_from_llm = self.query_llm(user_query, context)
            if llm_result is None:
                logger.error("LLM query returned None")
                answer, follow_up_questions = None, []
            else:
                answer, follow_up_questions = llm_result
            
            # Update sources from LLM call if available
            if sources_from_llm:
                result["sources"] = sources_from_llm
                result["sources_display"] = self.format_sources_for_display(sources_from_llm)
            
            result["answer"] = answer
            result["follow_up_questions"] = follow_up_questions
            result["answer_time"] = str(datetime.now() - start_time)
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
    
    def answer_query_energy_first(self, user_query: str, top_k: int = 20, 
                                   visualize: bool = True) -> Dict[str, Any]:
        logger.info("="*80)
        logger.info(f"Query: {user_query}")
        logger.info("ENERGY-FIRST PIPELINE")
        logger.info("="*80)
        
        # Step 1: Retrieve more candidates for better energy landscape
        passages, metadatas = self.retrieve_top_k(user_query, top_k=top_k)
        logger.info(f"[1/6] Retrieved {len(passages)} candidates")
        
        # Step 2: Rerank by semantic similarity
        ranked = self.rerank_passages(user_query, passages, metadatas)
        logger.info(f"[2/6] Reranked passages")
        
        # Step 3: Compute energy scores (THE CORE)
        ranked_passages = [p[0] for p in ranked]
        energies = self.compute_energy_scores(ranked_passages)
        logger.info(f"[3/6] 🔋 Computed energy landscape")
        
        # Combine all metrics
        passages_with_energy = []
        for (passage, metadata, similarity), energy in zip(ranked, energies):
            passages_with_energy.append({
                'passage': passage,
                'metadata': metadata,
                'similarity': similarity,
                'energy': energy,
                'quality_tier': self._classify_energy_tier(energy),
                'unified_score': 0.6 * similarity + 0.4 * (1 - energy)
            })
        
        # Step 4: ENERGY-BASED FILTERING (replaces arbitrary thresholds)
        high_quality, energy_threshold = self.energy_aware_filtering(
            ranked, energies, user_query
        )
        logger.info(f"[4/6] 🎯 Filtered to {len(high_quality)} high-quality passages "
                    f"(energy ≤ {energy_threshold:.3f})")
        
        # Step 5: ENERGY GATES (continuous confidence scoring)
        gate2_confidence, gate2_info = self.energy_gate_2_retrieval_quality(
            passages_with_energy, user_query
        )
        
        gate3_confidence, gate3_info = self.energy_gate_3_coherence(
            high_quality
        )
        
        # Combined gate confidence
        overall_confidence = (gate2_confidence + gate3_confidence) / 2
        
        logger.info(f"[5/6] 🚦 Energy Gates:")
        logger.info(f"      Gate 2 (Quality):   {gate2_confidence:.3f}")
        logger.info(f"      Gate 3 (Coherence): {gate3_confidence:.3f}")
        logger.info(f"      Overall:            {overall_confidence:.3f}")
        
        # Decision logic based on ENERGY confidence
        proceed_to_llm = overall_confidence >= 0.5
        
        if not proceed_to_llm:
            logger.warning(f"🛑 Energy confidence too low ({overall_confidence:.3f}) - rejecting query")
            
            return {
                "query": user_query,
                "status": "rejected_low_energy",
                "energy_confidence": overall_confidence,
                "gate2": gate2_info,
                "gate3": gate3_info,
                "energy_landscape": self.visualize_energy_landscape(
                    passages_with_energy, user_query
                ) if visualize else None,
                "recommendation": self._get_rejection_recommendation(
                    gate2_info, gate3_info
                ),
                "partial_evidence": high_quality[:3]  # Show what we found
            }
        
        # Step 6: Prepare context and query LLM
        filtered_passages = [
            (p['passage'], p['metadata'], p['similarity']) 
            for p in high_quality
        ]
        context = self.prepare_for_llm(filtered_passages)
        
        logger.info(f"[6/6] 💬 Querying LLM (confidence: {overall_confidence:.3f})")
        answer, follow_ups = self.query_llm(user_query, context)
        
        # Build result with ENERGY metrics prominently
        result = {
            "query": user_query,
            "status": "success",
            "answer": answer,
            "follow_up_questions": follow_ups,
            
            # ENERGY METRICS (the highlight!)
            "energy_confidence": round(overall_confidence, 3),
            "energy_landscape": self.visualize_energy_landscape(
                passages_with_energy, user_query
            ) if visualize else None,
            "energy_threshold": round(energy_threshold, 3),
            
            # Gate details
            "gates": {
                "gate2_quality": gate2_info,
                "gate3_coherence": gate3_info
            },
            
            # Evidence
            "evidence": {
                "total_retrieved": len(passages),
                "high_quality_count": len(high_quality),
                "quality_breakdown": {
                    'gold': sum(1 for p in passages_with_energy if p['quality_tier'] == 'GOLD'),
                    'silver': sum(1 for p in passages_with_energy if p['quality_tier'] == 'SILVER'),
                    'bronze': sum(1 for p in passages_with_energy if p['quality_tier'] == 'BRONZE'),
                    'weak': sum(1 for p in passages_with_energy if p['quality_tier'] == 'WEAK')
                },
                "passages": filtered_passages
            },
            
            # Context
            "context": context
        }
        
        return result
    
    def query_llm(self, user_query: str, context: str, model: str = "openai/gpt-4o-mini", 
                  temperature: float = None) -> tuple:
        if temperature is None:
            temperature = THRESHOLDS["llm_temperature"]
        
        # Handle both old and new prepare_for_llm return format
        if isinstance(context, tuple):
            context_text, sources_list = context
        else:
            context_text = context
            sources_list = []
            
        if self.enable_cache:
            cache_key = self._get_cache_key("llm", user_query, context_text, model, temperature)
            if cache_key in self.cache:
                self.cache_stats["llm_hits"] += 1
                cached_result = self.cache[cache_key]
                if cached_result is not None:
                    if isinstance(cached_result, tuple):
                        return cached_result, sources_list
                    else:
                        return (cached_result, []), sources_list
                else:
                    self.cache_stats["llm_misses"] += 1
            self.cache_stats["llm_misses"] += 1

        # Enhanced system prompt: warm, professional, efficient, and focused on clarity
        system_prompt = (
            "You are a warm, professional, and efficient medical assistant. Your goal is to provide clear, "
            "evidence-based answers that help users understand their health questions.\n\n"
            "PRINCIPLES:\n"
            "1. Be warm and supportive in tone - imagine speaking with a trusted healthcare provider\n"
            "2. Be precise and concise - avoid unnecessary jargon or lengthy explanations\n"
            "3. Cite sources using [1], [2], etc. notation - never invent information\n"
            "4. If uncertain, say so kindly and suggest where to learn more\n"
            "5. Focus on practical, actionable information the user can understand\n\n"
            "EVIDENCE-BASED FILTER:\n"
            "- Only use information from the provided sources\n"
            "- Do not add medical knowledge outside the given context\n"
            "- This prevents hallucinations while maintaining useful, direct answers\n\n"
            "OUTPUT FORMAT:\n"
            "- Start with a direct, warm answer\n"
            "- Use citations [1], [2], etc. for specific claims\n"
            "- Keep paragraphs short and scannable\n"
            "- End with follow-up questions the user might want to explore"
        )

        user_prompt = (
            f"QUESTION: {user_query}\n\n"
            f"EVIDENCE:\n{context_text}\n\n"
            f"Please provide a warm, clear, and professional answer based only on the evidence above.\n"
            f"Cite sources as [1], [2], etc. when making specific claims.\n"
            f"If you cannot answer with confidence, explain why gently and suggest what information would help.\n\n"
            f"After your answer, provide 3-4 follow-up questions the user might want to explore next."
        )

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
            return (None, []), sources_list

        # Extract follow-up questions if present
        follow_up_questions = []
        if answer:
            import re
            # Look for numbered list at the end
            follow_up_match = re.search(
                r'(?:follow-?up|next|explore|might want|could ask|further)\s*(?:questions?)?:?\s*(.+?)$',
                answer,
                re.IGNORECASE | re.DOTALL
            )
            if follow_up_match:
                questions_text = follow_up_match.group(1)
                question_matches = re.findall(r'^\s*\d+\.\s*(.+?)$', questions_text, re.MULTILINE)
                follow_up_questions = [q.strip() for q in question_matches if q.strip()]

        result = (answer, follow_up_questions)

        if self.enable_cache:
            self.cache.set(cache_key, result, expire=self.cache_ttl)

        return result, sources_list
    
    def format_sources_for_display(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources list for user-friendly display using LangChain toolcall format.
        
        Args:
            sources: List of source dictionaries from prepare_for_llm
            
        Returns:
            Formatted string of sources with citations and links
        """
        if not sources:
            return ""
        
        output = ["## Sources"]
        for source in sources:
            citation = source.get("citation_number", "?")
            title = source.get("title", "Unknown")
            year = source.get("year", "")
            source_type = source.get("source_type", "")
            pmid = source.get("pmid", "")
            url = source.get("url", "")
            
            # Build source line
            source_line = f"[{citation}] {title}"
            if year and year != "Unknown":
                source_line += f" ({year})"
            if source_type and source_type != "Unknown":
                source_line += f" - {source_type}"
            
            # Add link if available
            if pmid:
                source_line += f" - https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            elif url:
                source_line += f" - {url}"
            
            output.append(source_line)
        
        return "\n".join(output)
    
    def _get_rejection_recommendation(self, gate2_info, gate3_info):
        recommendations = []
        
        # Diagnose gate 2 (quality) failures
        if gate2_info['confidence'] < 0.5:
            if gate2_info['gold_tier_count'] == 0:
                recommendations.append(
                    "No high-quality evidence found. "
                    "Try reformulating with standard medical terminology."
                )
            if gate2_info['energy_min'] > 0.5:
                recommendations.append(
                    "All evidence has high energy (low quality). "
                    "This topic may not be well-covered in the knowledge base."
                )
        
        # Diagnose gate 3 (coherence) failures  
        if gate3_info['confidence'] < 0.5:
            if gate3_info['num_clusters'] > 3:
                recommendations.append(
                    "Evidence is fragmented across multiple quality levels. "
                    "Sources may be discussing different aspects or have conflicting quality."
                )
        
        if not recommendations:
            recommendations.append(
                "Consider consulting primary literature or clinical guidelines."
            )
        
        return recommendations
    
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
            
            # Display follow-up questions if available
            if result.get("follow_up_questions"):
                print("\n### Explore Further:")
                for i, q in enumerate(result["follow_up_questions"], 1):
                    print(f"  {i}. {q}")
        else:
            print("LLM skipped; context prepared.")
        
        # Display sources with citations (using LangChain format)
        if result.get("sources_display"):
            print("\n" + "="*80)
            print(result["sources_display"])
            print("="*80)
        
        # Display evidence chain verification if performed
        if result.get("evidence_chain"):
            print(RetrievalPipeline.format_evidence_chain(result["evidence_chain"]))



if __name__ == "__main__":
    main()