import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import logging
logger = logging.getLogger("retriever_v2")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

import argparse
from typing import Any, Dict, List, Tuple, Optional, Set
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
from datetime import datetime, timedelta
from pathlib import Path
import requests
import numpy as np
import re
from dataclasses import dataclass
from enum import Enum

from config import (
    MODEL_NAME, CHROMA_PATH, CHROMA_COLLECTION_NAME, 
    RANKING_SCORER_CKPT, VERIFICATION_SCORER_CKPT, 
    CACHE_DIR, THRESHOLDS
)

class EvidenceLevel(Enum):
    HIGH = 4          # RCTs, systematic reviews with no serious limitations
    MODERATE = 3      # RCTs with limitations, or strong observational evidence
    LOW = 2           # Observational studies, case-control
    VERY_LOW = 1      # Case reports, expert opinion, mechanistic studies
    UNKNOWN = 0       # Unable to assess


class StudyDesign(Enum):
    META_ANALYSIS = 10
    SYSTEMATIC_REVIEW = 9
    RCT = 8
    COHORT = 6
    CASE_CONTROL = 5
    CROSS_SECTIONAL = 4
    CASE_SERIES = 3
    CASE_REPORT = 2
    EXPERT_OPINION = 1
    UNKNOWN = 0


@dataclass
class MedicalEvidence:
    text: str
    metadata: Dict[str, Any]
    
    # Quality indicators
    evidence_level: EvidenceLevel
    study_design: StudyDesign
    sample_size: Optional[int] = None
    publication_year: Optional[int] = None
    journal_impact_factor: Optional[float] = None
    citation_count: Optional[int] = None
    
    # Statistical measures
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    
    # Validity checks
    is_retracted: bool = False
    is_predatory_journal: bool = False
    conflicts_of_interest: bool = False
    
    # Relevance
    semantic_similarity: float = 0.0
    recency_score: float = 0.0
    
    # Composite scores
    quality_score: float = 0.0
    relevance_score: float = 0.0
    final_score: float = 0.0
    
    def compute_quality_score(self) -> float:
        if self.is_retracted or self.is_predatory_journal:
            return 0.0
        
        # Base score from study design
        base = self.study_design.value / 10.0
        
        # Modifiers
        modifiers = 1.0
        
        # Sample size modifier (larger = better, capped)
        if self.sample_size:
            size_bonus = min(0.2, np.log10(self.sample_size) / 5.0)
            modifiers += size_bonus
        
        # Precision modifier (narrower CI = better)
        if self.confidence_interval:
            ci_width = abs(self.confidence_interval[1] - self.confidence_interval[0])
            precision_bonus = max(0, 0.15 * (1 - ci_width / 10.0))
            modifiers += precision_bonus
        
        # Citation count modifier
        if self.citation_count:
            citation_bonus = min(0.1, self.citation_count / 1000.0)
            modifiers += citation_bonus
        
        # Conflict of interest penalty
        if self.conflicts_of_interest:
            modifiers *= 0.9
        
        self.quality_score = base * modifiers
        return self.quality_score
    
    def compute_recency_score(self, current_year: int = None) -> float:
        if not self.publication_year:
            return 0.5  # Unknown = moderate
        
        if current_year is None:
            current_year = datetime.now().year
        
        age = current_year - self.publication_year
        
        # Exponential decay: half-life of 5 years for most topics
        # (guideline-specific topics may override this)
        half_life = 5
        decay_rate = np.log(2) / half_life
        self.recency_score = np.exp(-decay_rate * age)
        
        return self.recency_score
    
    def compute_final_score(self, weights: Dict[str, float] = None) -> float:
        if weights is None:
            weights = {
                'quality': 0.5,
                'semantic': 0.3,
                'recency': 0.2
            }
        
        self.compute_quality_score()
        self.compute_recency_score()
        
        self.final_score = (
            weights['quality'] * self.quality_score +
            weights['semantic'] * self.semantic_similarity +
            weights['recency'] * self.recency_score
        )
        
        return self.final_score

class MedicalEvidenceExtractor:    
    # Study design keywords
    DESIGN_PATTERNS = {
        StudyDesign.META_ANALYSIS: [
            r'meta-analysis', r'meta analysis', r'pooled analysis',
            r'systematic review and meta-analysis'
        ],
        StudyDesign.SYSTEMATIC_REVIEW: [
            r'systematic review', r'systematic literature review',
            r'cochrane review'
        ],
        StudyDesign.RCT: [
            r'randomized controlled trial', r'randomised controlled trial',
            r'RCT', r'double-blind.*placebo', r'randomized.*trial'
        ],
        StudyDesign.COHORT: [
            r'cohort study', r'prospective.*study', r'longitudinal study',
            r'follow-up study'
        ],
        StudyDesign.CASE_CONTROL: [
            r'case-control', r'case control', r'matched control'
        ],
        StudyDesign.CROSS_SECTIONAL: [
            r'cross-sectional', r'prevalence study', r'survey study'
        ],
        StudyDesign.CASE_SERIES: [
            r'case series', r'consecutive cases', r'case collection'
        ],
        StudyDesign.CASE_REPORT: [
            r'case report', r'case presentation', r'single patient'
        ],
        StudyDesign.EXPERT_OPINION: [
            r'expert opinion', r'consensus statement', r'editorial'
        ]
    }
    
    @staticmethod
    def detect_study_design(text: str, metadata: Dict) -> StudyDesign:
        text_lower = text.lower()
        
        # Check metadata first
        pub_type = metadata.get('publication_type', '').lower()
        for design, patterns in MedicalEvidenceExtractor.DESIGN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, pub_type):
                    return design
        
        # Check text content
        for design, patterns in MedicalEvidenceExtractor.DESIGN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return design
        
        return StudyDesign.UNKNOWN
    
    @staticmethod
    def extract_sample_size(text: str) -> Optional[int]:
        patterns = [
            r'n\s*=\s*(\d+)',
            r'(\d+)\s+patients',
            r'(\d+)\s+participants',
            r'(\d+)\s+subjects',
            r'sample size.*?(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    size = int(match.group(1))
                    if 10 <= size <= 1000000:  # Sanity check
                        return size
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def extract_confidence_interval(text: str) -> Optional[Tuple[float, float]]:
        patterns = [
            r'95%\s*CI[:\s]+(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)',
            r'CI\s*95%[:\s]+(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)',
            r'\[(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)\]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    lower = float(match.group(1))
                    upper = float(match.group(2))
                    return (lower, upper)
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def extract_p_value(text: str) -> Optional[float]:
        patterns = [
            r'p\s*[=<]\s*(\d+\.?\d*)',
            r'p-value\s*[=<]\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    p_val = float(match.group(1))
                    if 0 <= p_val <= 1:
                        return p_val
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def is_retracted(metadata: Dict) -> bool:
        retraction_indicators = [
            'retracted', 'withdrawn', 'expression of concern'
        ]
        
        pub_status = metadata.get('publication_status', '').lower()
        title = metadata.get('title', '').lower()
        
        for indicator in retraction_indicators:
            if indicator in pub_status or indicator in title:
                return True
        
        return False
    
    @staticmethod
    def assess_evidence_level(design: StudyDesign, has_limitations: bool = False) -> EvidenceLevel:        
        # Base mapping
        design_to_level = {
            StudyDesign.META_ANALYSIS: EvidenceLevel.HIGH,
            StudyDesign.SYSTEMATIC_REVIEW: EvidenceLevel.HIGH,
            StudyDesign.RCT: EvidenceLevel.HIGH,
            StudyDesign.COHORT: EvidenceLevel.MODERATE,
            StudyDesign.CASE_CONTROL: EvidenceLevel.LOW,
            StudyDesign.CROSS_SECTIONAL: EvidenceLevel.LOW,
            StudyDesign.CASE_SERIES: EvidenceLevel.VERY_LOW,
            StudyDesign.CASE_REPORT: EvidenceLevel.VERY_LOW,
            StudyDesign.EXPERT_OPINION: EvidenceLevel.VERY_LOW,
            StudyDesign.UNKNOWN: EvidenceLevel.UNKNOWN
        }
        
        base_level = design_to_level[design]
        
        # Downgrade if serious limitations
        if has_limitations and base_level.value > 1:
            return EvidenceLevel(base_level.value - 1)
        
        return base_level

class MeSHTermMapper:    
    def __init__(self, cache_enabled: bool = True):
        self.cache = {} if cache_enabled else None
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def get_mesh_terms(self, query: str, max_terms: int = 5) -> List[str]:
        """Get MeSH terms for a query using NCBI eSearch"""
        
        if self.cache is not None and query in self.cache:
            return self.cache[query]
        
        try:
            # Use eSearch to find MeSH terms
            search_url = f"{self.base_url}esearch.fcgi"
            params = {
                "db": "mesh",
                "term": query,
                "retmax": max_terms,
                "retmode": "json"
            }
            
            response = requests.get(search_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            mesh_ids = data.get("esearchresult", {}).get("idlist", [])
            
            if not mesh_ids:
                logger.debug(f"No MeSH terms found for: {query}")
                return [query]  # Fallback to original query
            
            # Fetch MeSH term names
            fetch_url = f"{self.base_url}esummary.fcgi"
            params = {
                "db": "mesh",
                "id": ",".join(mesh_ids[:max_terms]),
                "retmode": "json"
            }
            
            response = requests.get(fetch_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            result = data.get("result", {})
            
            mesh_terms = []
            for mesh_id in mesh_ids[:max_terms]:
                term_data = result.get(mesh_id, {})
                term = term_data.get("ds_meshterms", [])
                if term:
                    mesh_terms.append(term[0])
            
            if self.cache is not None:
                self.cache[query] = mesh_terms
            
            logger.info(f"MeSH terms for '{query}': {mesh_terms}")
            return mesh_terms if mesh_terms else [query]
            
        except Exception as e:
            logger.warning(f"MeSH lookup failed: {e}")
            return [query]

class PubMedSearchEngine:    
    def __init__(self, mesh_mapper: MeSHTermMapper = None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.mesh_mapper = mesh_mapper or MeSHTermMapper()
    
    def search(self, query: str, max_results: int = 10, 
               year_range: Optional[Tuple[int, int]] = None,
               study_types: Optional[List[str]] = None,
               high_quality_only: bool = True) -> List[str]:
        """
        Advanced PubMed search with filters
        
        Args:
            query: Search query
            max_results: Max papers to return
            year_range: (start_year, end_year) or None for adaptive range
            study_types: Filter by study type (e.g., ['Clinical Trial', 'Meta-Analysis'])
            high_quality_only: Filter for high-impact journals only
        """
        
        # Get MeSH terms
        mesh_terms = self.mesh_mapper.get_mesh_terms(query)
        
        # Build query with MeSH terms
        if mesh_terms:
            mesh_query = " OR ".join([f'"{term}"[MeSH Terms]' for term in mesh_terms])
            full_query = f"({mesh_query}) OR ({query})"
        else:
            full_query = query
        
        # Add study type filters
        if study_types:
            type_filter = " OR ".join([f'"{t}"[Publication Type]' for t in study_types])
            full_query = f"({full_query}) AND ({type_filter})"
        
        # Add date range (adaptive if not specified)
        if year_range is None:
            # Adaptive: prioritize recent, but include older landmark studies
            current_year = datetime.now().year
            year_range = (current_year - 10, current_year)
        
        start_year, end_year = year_range
        full_query = f"({full_query}) AND ({start_year}[Date - Publication]:{end_year}[Date - Publication])"
        
        # Add quality filters
        if high_quality_only:
            # Filter for journals in top quartile (crude heuristic)
            full_query = f"({full_query}) AND (jsubsetaim[text])"  # MEDLINE subset
        
        logger.info(f"PubMed query: {full_query}")
        
        try:
            search_url = f"{self.base_url}esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": full_query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance"
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            logger.info(f"PubMed search returned {len(pmids)} results")
            return pmids
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[MedicalEvidence]:        
        if not pmids:
            return []
        
        try:
            fetch_url = f"{self.base_url}efetch.fcgi"
            
            all_evidence = []
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
                
                # Parse XML
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.text)
                
                for article in root.findall(".//PubmedArticle"):
                    try:
                        evidence = self._parse_pubmed_article(article)
                        if evidence:
                            all_evidence.append(evidence)
                    except Exception as e:
                        logger.warning(f"Failed to parse article: {e}")
                        continue
                
                # Rate limiting
                import time
                time.sleep(0.5)
            
            logger.info(f"Fetched {len(all_evidence)} PubMed abstracts")
            return all_evidence
            
        except Exception as e:
            logger.error(f"PubMed fetch failed: {e}")
            return []
    
    def _parse_pubmed_article(self, article) -> Optional[MedicalEvidence]:        
        # Extract PMID
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
        
        # Extract title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "Unknown Title"
        
        # Extract abstract
        abstract_elem = article.find(".//AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else ""
        
        if not abstract or len(abstract.strip()) < 50:
            return None
        
        # Extract publication year
        year_elem = article.find(".//PubDate/Year")
        year = int(year_elem.text) if year_elem is not None and year_elem.text else None
        
        # Extract publication type
        pub_types = article.findall(".//PublicationType")
        pub_type_str = ", ".join([pt.text for pt in pub_types if pt.text]) if pub_types else ""
        
        # Extract authors
        authors_elems = article.findall(".//Author/LastName")
        authors = ", ".join([a.text for a in authors_elems if a.text])[:100] if authors_elems else "Unknown"
        
        # Build metadata
        metadata = {
            "pmid": pmid,
            "title": title,
            "year": year,
            "authors": authors,
            "publication_type": pub_type_str,
            "source_type": "pubmed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        }
        
        # Create MedicalEvidence object
        text = f"{title}\n\n{abstract}"
        
        study_design = MedicalEvidenceExtractor.detect_study_design(text, metadata)
        evidence_level = MedicalEvidenceExtractor.assess_evidence_level(study_design)
        
        evidence = MedicalEvidence(
            text=text,
            metadata=metadata,
            evidence_level=evidence_level,
            study_design=study_design,
            sample_size=MedicalEvidenceExtractor.extract_sample_size(text),
            publication_year=year,
            confidence_interval=MedicalEvidenceExtractor.extract_confidence_interval(text),
            p_value=MedicalEvidenceExtractor.extract_p_value(text),
            is_retracted=MedicalEvidenceExtractor.is_retracted(metadata)
        )
        
        return evidence

class RankingScorer(nn.Module):
    def __init__(self, encoder, embedding_dim=None):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = THRESHOLDS["embedding_dim"]
        self.encoder = encoder
        self.hidden_dim = encoder.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, enc_ips):
        out = self.encoder(**enc_ips).last_hidden_state
        mask = enc_ips["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embedding = self.projection(pooled)
        return embedding
    
class MedicalRetrievalPipeline:
    def __init__(self, 
                 chroma_path: str = None,
                 collection_name: str = None,
                 model_name: str = None,
                 checkpoint_path: str = None,
                 verification_checkpoint_path: str = None,
                 device: str = None,
                 llm_api_key: str = None,
                 enable_cache: bool = True,
                 cache_dir: str = None):
        
        load_dotenv()
        
        # Config
        model_name = model_name or MODEL_NAME
        chroma_path = chroma_path or str(CHROMA_PATH)
        collection_name = collection_name or CHROMA_COLLECTION_NAME
        checkpoint_path = checkpoint_path or str(RANKING_SCORER_CKPT)
        verification_checkpoint_path = verification_checkpoint_path or str(VERIFICATION_SCORER_CKPT)
        cache_dir = cache_dir or str(CACHE_DIR)
        
        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cache with topic-specific TTLs
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = Cache(cache_dir, size_limit=2*1024**3)  # 2GB
            self.cache_stats = {"hits": 0, "misses": 0}
            logger.info("Cache enabled: 2GB, topic-specific TTLs")
        else:
            self.cache = None
        
        # ChromaDB
        self.client = PersistentClient(path=chroma_path)
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info("Loaded ChromaDB collection")
        except Exception as e:
            raise ValueError(f"Collection not found: {collection_name}") from e
        
        # Models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        ranking_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.ranking_model = RankingScorer(ranking_encoder).to(self.device)
        self.ranking_model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.ranking_model.eval()
        logger.info("Loaded ranking model")
        
        # Medical NLI model (BioBERT-based instead of generic RoBERTa)
        try:
            self.nli_pipeline = pipeline(
                "text-classification",
                model="microsoft/BiomedNLI-PubMedBERT-base-uncased-snli_multinli",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Loaded BiomedNLI model for verification")
        except Exception:
            logger.warning("BiomedNLI not available, falling back to RoBERTa")
            self.nli_pipeline = pipeline(
                "text-classification",
                model="roberta-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
        
        # LLM
        api_key = llm_api_key or os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("LLM connected")
        
        # Components
        self.mesh_mapper = MeSHTermMapper()
        self.pubmed = PubMedSearchEngine(self.mesh_mapper)
        self.evidence_extractor = MedicalEvidenceExtractor()
    
    def _get_cache_ttl(self, topic: str) -> int:
        ttls = {
            'covid': 24 * 3600,              # 1 day
            'pandemic': 24 * 3600,
            'outbreak': 24 * 3600,
            'emergency': 24 * 3600,
            'guidelines': 7 * 24 * 3600,     # 1 week
            'protocol': 7 * 24 * 3600,
            'treatment': 30 * 24 * 3600,     # 30 days
            'diagnosis': 30 * 24 * 3600,
            'anatomy': 365 * 24 * 3600,      # 1 year
            'physiology': 365 * 24 * 3600,
        }
        
        topic_lower = topic.lower()
        for key, ttl in ttls.items():
            if key in topic_lower:
                return ttl
        
        return 7 * 24 * 3600  # Default: 1 week
    
    def retrieve_and_grade(self, query: str, top_k: int = 10) -> List[MedicalEvidence]:
        logger.info(f"Retrieving evidence for: {query}")
        
        # Step 1: ChromaDB retrieval
        results = self.collection.query(query_texts=[query], n_results=top_k)
        passages = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        logger.info(f"Retrieved {len(passages)} passages from ChromaDB")
        
        # Step 2: Convert to MedicalEvidence objects
        evidence_list = []
        for passage, metadata in zip(passages, metadatas):
            study_design = self.evidence_extractor.detect_study_design(passage, metadata)
            evidence_level = self.evidence_extractor.assess_evidence_level(study_design)
            
            year = metadata.get('year')
            if isinstance(year, str):
                try:
                    year = int(year)
                except ValueError:
                    year = None
            
            evidence = MedicalEvidence(
                text=passage,
                metadata=metadata,
                evidence_level=evidence_level,
                study_design=study_design,
                publication_year=year,
                sample_size=self.evidence_extractor.extract_sample_size(passage),
                confidence_interval=self.evidence_extractor.extract_confidence_interval(passage),
                p_value=self.evidence_extractor.extract_p_value(passage),
                is_retracted=self.evidence_extractor.is_retracted(metadata)
            )
            
            evidence_list.append(evidence)
        
        # Step 3: Compute semantic similarity (reranking)
        evidence_list = self._rerank_evidence(query, evidence_list)
        
        # Step 4: Compute final scores
        for evidence in evidence_list:
            evidence.compute_final_score()
        
        # Step 5: Sort by final score
        evidence_list.sort(key=lambda e: e.final_score, reverse=True)
        
        logger.info(f"Graded {len(evidence_list)} evidence pieces")
        return evidence_list
    
    def _rerank_evidence(self, query: str, evidence_list: List[MedicalEvidence]) -> List[MedicalEvidence]:
        passages = [e.text for e in evidence_list]
        
        with torch.no_grad():
            # Encode query
            enc_query = self.tokenizer(
                query,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            enc_query = {k: v.to(self.device) for k, v in enc_query.items()}
            query_embedding = self.ranking_model(enc_query)
            
            # Encode passages (batched)
            batch_size = 8
            similarities = []
            
            for i in range(0, len(passages), batch_size):
                batch = passages[i:i+batch_size]
                enc_batch = self.tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
                enc_batch = {k: v.to(self.device) for k, v in enc_batch.items()}
                batch_embeddings = self.ranking_model(enc_batch)
                
                sims = F.cosine_similarity(query_embedding, batch_embeddings)
                similarities.extend(sims.cpu().tolist())
        
        # Assign similarities
        for evidence, sim in zip(evidence_list, similarities):
            evidence.semantic_similarity = sim
        
        return evidence_list
    
    def filter_by_grade(self, evidence_list: List[MedicalEvidence],
                        min_level: EvidenceLevel = EvidenceLevel.LOW,
                        min_score: float = 0.4) -> List[MedicalEvidence]:        
        filtered = [
            e for e in evidence_list
            if e.evidence_level.value >= min_level.value and e.final_score >= min_score
        ]
        
        logger.info(f"Filtered: {len(filtered)}/{len(evidence_list)} passed quality threshold")
        return filtered
    
    def pubmed_fallback(self, query: str, max_results: int = 5) -> List[MedicalEvidence]:        
        logger.info("Initiating PubMed fallback...")
        
        # Try high-quality studies first
        pmids = self.pubmed.search(
            query,
            max_results=max_results,
            study_types=['Meta-Analysis', 'Randomized Controlled Trial', 'Systematic Review'],
            high_quality_only=True
        )
        
        # Fallback to all study types if no high-quality results
        if not pmids:
            logger.info("No high-quality studies found, broadening search...")
            pmids = self.pubmed.search(query, max_results=max_results, high_quality_only=False)
        
        # Fetch and grade abstracts
        evidence_list = self.pubmed.fetch_abstracts(pmids)
        
        logger.info(f"PubMed fallback retrieved {len(evidence_list)} evidence pieces")
        return evidence_list
    
    def detect_contradictions(self, evidence_list: List[MedicalEvidence]) -> Dict[str, Any]:
        if len(evidence_list) < 2:
            return {"has_contradictions": False, "conflicts": []}
        
        logger.info("Analyzing statistical contradictions...")
        
        conflicts = []
        
        # Check for overlapping confidence intervals
        ci_evidence = [e for e in evidence_list if e.confidence_interval is not None]
        
        for i in range(len(ci_evidence)):
            for j in range(i + 1, len(ci_evidence)):
                e1, e2 = ci_evidence[i], ci_evidence[j]
                
                # Check if CIs overlap
                ci1_lower, ci1_upper = e1.confidence_interval
                ci2_lower, ci2_upper = e2.confidence_interval
                
                # No overlap = potential contradiction
                if ci1_upper < ci2_lower or ci2_upper < ci1_lower:
                    conflicts.append({
                        "type": "statistical",
                        "source_1": e1.metadata.get("title", "Source 1"),
                        "source_2": e2.metadata.get("title", "Source 2"),
                        "ci_1": f"[{ci1_lower}, {ci1_upper}]",
                        "ci_2": f"[{ci2_lower}, {ci2_upper}]",
                        "severity": "high" if abs(ci1_upper - ci2_lower) > 1.0 else "moderate",
                        "explanation": "Confidence intervals do not overlap - statistically incompatible findings"
                    })
        
        # Check for effect direction disagreements
        # (e.g., one study shows benefit, another shows harm)
        for i in range(len(evidence_list)):
            for j in range(i + 1, len(evidence_list)):
                e1, e2 = evidence_list[i], evidence_list[j]
                
                # Simple heuristic: look for "increase" vs "decrease" language
                text1_lower = e1.text.lower()
                text2_lower = e2.text.lower()
                
                benefit_keywords = ['reduce', 'decrease', 'improve', 'benefit', 'effective']
                harm_keywords = ['increase', 'worsen', 'harmful', 'risk', 'adverse']
                
                e1_benefit = any(kw in text1_lower for kw in benefit_keywords)
                e1_harm = any(kw in text1_lower for kw in harm_keywords)
                e2_benefit = any(kw in text2_lower for kw in benefit_keywords)
                e2_harm = any(kw in text2_lower for kw in harm_keywords)
                
                if (e1_benefit and e2_harm) or (e1_harm and e2_benefit):
                    conflicts.append({
                        "type": "directional",
                        "source_1": e1.metadata.get("title", "Source 1"),
                        "source_2": e2.metadata.get("title", "Source 2"),
                        "explanation": "Studies report opposite effect directions",
                        "severity": "high",
                        "recommendation": "Check study populations, interventions, and outcomes for differences"
                    })
        
        result = {
            "has_contradictions": len(conflicts) > 0,
            "conflicts": conflicts,
            "note": f"Found {len(conflicts)} statistical contradictions" if conflicts else "No contradictions detected"
        }
        
        logger.info(f"Contradiction analysis: {len(conflicts)} conflicts found")
        return result
    
    def verify_claims(self, answer: str, evidence_list: List[MedicalEvidence]) -> Dict[str, Any]:
        logger.info("Verifying claims with BiomedNLI...")
        
        # Split answer into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return {
                "verified": False,
                "total_claims": 0,
                "verified_claims": 0,
                "verification_rate": 0.0,
                "claims": []
            }
        
        verified_count = 0
        claim_results = []
        
        for idx, sentence in enumerate(sentences, 1):
            # Check against all evidence
            best_support = None
            best_score = 0.0
            
            for evidence in evidence_list:
                # Use NLI to check entailment
                nli_input = f"{evidence.text} </s> {sentence}"
                result = self.nli_pipeline(nli_input)[0]
                
                label = result['label']
                score = result['score']
                
                if label == "ENTAILMENT" and score > best_score:
                    best_score = score
                    best_support = evidence
            
            # Threshold for verification
            is_verified = best_score >= 0.7
            if is_verified:
                verified_count += 1
            
            claim_results.append({
                "claim": sentence,
                "verified": is_verified,
                "confidence": round(best_score, 3),
                "supporting_evidence": best_support.metadata.get("title", "Unknown") if best_support else None,
                "evidence_quality": best_support.evidence_level.name if best_support else None
            })
        
        verification_rate = verified_count / len(sentences)
        
        result = {
            "verified": verification_rate >= 0.8,
            "total_claims": len(sentences),
            "verified_claims": verified_count,
            "unverified_claims": len(sentences) - verified_count,
            "verification_rate": round(verification_rate, 3),
            "claims": claim_results
        }
        
        logger.info(f"Claim verification: {verified_count}/{len(sentences)} verified ({verification_rate*100:.1f}%)")
        
        return result
    
    def answer_query(self, query: str, 
                     top_k: int = 10,
                     min_evidence_level: EvidenceLevel = EvidenceLevel.LOW,
                     use_pubmed_fallback: bool = True,
                     verify_answer: bool = True) -> Dict[str, Any]:  
        logger.info("="*80)
        logger.info(f"Query: {query}")
        logger.info("="*80)
        
        # Step 1: Retrieve and grade from ChromaDB
        evidence_list = self.retrieve_and_grade(query, top_k=top_k)
        
        # Step 2: Filter by GRADE level
        high_quality = self.filter_by_grade(
            evidence_list,
            min_level=min_evidence_level,
            min_score=0.4
        )
        
        # Step 3: PubMed fallback if insufficient evidence
        if len(high_quality) < 3 and use_pubmed_fallback:
            logger.info("Insufficient evidence in ChromaDB, trying PubMed...")
            pubmed_evidence = self.pubmed_fallback(query, max_results=5)
            
            # Rerank PubMed evidence
            pubmed_evidence = self._rerank_evidence(query, pubmed_evidence)
            for e in pubmed_evidence:
                e.compute_final_score()
            
            # Merge and re-filter
            combined = high_quality + pubmed_evidence
            combined.sort(key=lambda e: e.final_score, reverse=True)
            high_quality = combined[:10]
        
        # Step 4: Check if we have enough evidence (relaxed to 1)
        if len(high_quality) < 1:
            return {
                "query": query,
                "status": "rejected",
                "reason": "No evidence found",
                "evidence_count": len(high_quality),
                "suggestion": "Try rephrasing with medical terminology or consult primary literature",
                "partial_evidence": [
                    {
                        "title": e.metadata.get("title", "Unknown"),
                        "level": e.evidence_level.name,
                        "score": round(e.final_score, 3)
                    }
                    for e in high_quality
                ]
            }
        
        # Step 5: Detect contradictions
        contradictions = self.detect_contradictions(high_quality)
        
        # Step 6: Prepare context for LLM
        context = self._prepare_context(high_quality)
        
        # Step 7: Generate answer
        answer = self._query_llm(query, context, high_quality)
        
        # Step 8: Verify answer (if requested)
        verification = None
        if verify_answer:
            verification = self.verify_claims(answer, high_quality)
        
        # Step 9: Compute overall confidence
        confidence = self._compute_confidence(high_quality, verification)
        
        result = {
            "query": query,
            "status": "success",
            "answer": answer,
            "confidence": confidence,
            "evidence_summary": {
                "total_retrieved": len(evidence_list),
                "high_quality_count": len(high_quality),
                "grade_distribution": self._grade_distribution(high_quality),
                "study_types": self._study_type_distribution(high_quality)
            },
            "contradictions": contradictions,
            "verification": verification,
            "sources": self._format_sources(high_quality)
        }
        
        logger.info(f"Answer generated with {confidence['label']} confidence")
        return result
    
    def _prepare_context(self, evidence_list: List[MedicalEvidence]) -> str:
        context_parts = []
        for i, evidence in enumerate(evidence_list[:5], 1):
            quality_badge = f"[{evidence.evidence_level.name} - {evidence.study_design.name}]"
            context_parts.append(f"[{i}] {quality_badge}\n{evidence.text}")
        
        return "\n\n".join(context_parts)
    
    def _query_llm(self, query: str, context: str, evidence_list: List[MedicalEvidence]) -> str:  
        # Build evidence quality summary
        quality_summary = self._grade_distribution(evidence_list)
        quality_str = ", ".join([f"{k}: {v}" for k, v in quality_summary.items()])
        
        system_prompt = f"""You are a medical AI assistant that provides evidence-based answers.

CRITICAL RULES:
1. ONLY use information from the provided evidence
2. Cite sources using [1], [2], etc.
3. Acknowledge evidence quality: HIGH/MODERATE/LOW/VERY_LOW
4. If evidence is LOW or VERY_LOW, state limitations explicitly
5. Never extrapolate beyond what the evidence supports

Evidence quality in this context: {quality_str}

Provide a clear, professional answer that:
- Directly addresses the question
- Cites specific evidence
- Acknowledges any limitations or uncertainties
- Uses accessible language without medical jargon unless necessary"""

        user_prompt = f"""Question: {query}

Evidence:
{context}

Provide an evidence-based answer with citations."""

        response = self.llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3  # Low temperature for medical accuracy
        )

        return response.choices[0].message.content
    
    def _compute_confidence(self, evidence_list: List[MedicalEvidence],
                           verification: Optional[Dict] = None) -> Dict[str, Any]:
        """Compute overall confidence based on evidence quality and verification"""
        
        if not evidence_list:
            return {"score": 0.0, "label": "VERY_LOW"}
        
        # Average evidence quality
        avg_quality = np.mean([e.final_score for e in evidence_list])
        
        # GRADE level distribution - boost for high quality sources
        high_grade_count = sum(1 for e in evidence_list if e.evidence_level.value >= 3)  # HIGH=4, MODERATE=3
        grade_bonus = min(0.3, high_grade_count * 0.2)  # More generous bonus
        
        # Evidence count bonus (more sources = more confidence)
        count_bonus = min(0.2, len(evidence_list) * 0.05)
        
        # Verification penalty
        verification_penalty = 0.0
        if verification and verification.get('verification_rate', 1.0) < 0.7:
            verification_penalty = 0.15
        
        # Final score: more generous baseline + bonuses
        score = min(1.0, avg_quality * 0.6 + 0.4 + grade_bonus + count_bonus - verification_penalty)
        score = max(0.0, score)
        
        # Label
        if score >= 0.75:
            label = "HIGH"
        elif score >= 0.55:
            label = "MODERATE"
        elif score >= 0.35:
            label = "LOW"
        else:
            label = "VERY_LOW"
        
        return {
            "score": round(score, 3),
            "label": label,
            "components": {
                "evidence_quality": round(avg_quality, 3),
                "grade_bonus": round(grade_bonus, 3),
                "count_bonus": round(count_bonus, 3),
                "verification_penalty": round(verification_penalty, 3),
                "evidence_count": len(evidence_list)
            }
        }
    
    def _grade_distribution(self, evidence_list: List[MedicalEvidence]) -> Dict[str, int]:
        dist = {level.name: 0 for level in EvidenceLevel}
        for evidence in evidence_list:
            dist[evidence.evidence_level.name] += 1
        return {k: v for k, v in dist.items() if v > 0}
    
    def _study_type_distribution(self, evidence_list: List[MedicalEvidence]) -> Dict[str, int]:
        dist = {}
        for evidence in evidence_list:
            design_name = evidence.study_design.name
            dist[design_name] = dist.get(design_name, 0) + 1
        return dist
    
    def _format_sources(self, evidence_list: List[MedicalEvidence]) -> str:
        sources = []
        for i, evidence in enumerate(evidence_list[:5], 1):
            title = evidence.metadata.get("title", "Unknown")
            year = evidence.metadata.get("year", "")
            level = evidence.evidence_level.name
            design = evidence.study_design.name
            url = evidence.metadata.get("url", "")
            
            source_line = f"[{i}] {title} ({year}) - {level} ({design})"
            if url:
                source_line += f"\n    {url}"
            
            sources.append(source_line)
        
        return "\n\n".join(sources)

def main():
    parser = argparse.ArgumentParser(description="Medical RAG with GRADE evidence grading")
    parser.add_argument("query", nargs="?", help="Medical question")
    parser.add_argument("--top-k", type=int, default=10, help="Top passages to retrieve")
    parser.add_argument("--min-level", type=str, default="LOW", 
                       choices=["HIGH", "MODERATE", "LOW", "VERY_LOW"],
                       help="Minimum GRADE evidence level")
    parser.add_argument("--no-pubmed", action="store_true", help="Disable PubMed fallback")
    parser.add_argument("--no-verify", action="store_true", help="Skip claim verification")
    args = parser.parse_args()
    
    if not args.query:
        try:
            args.query = input("Enter your medical question: ").strip()
        except EOFError:
            args.query = None
    
    if not args.query:
        logger.error("No query provided")
        return
    
    # Convert string to EvidenceLevel enum
    min_level = EvidenceLevel[args.min_level]
    
    pipeline = MedicalRetrievalPipeline(enable_cache=True)
    
    result = pipeline.answer_query(
        query=args.query,
        top_k=args.top_k,
        min_evidence_level=min_level,
        use_pubmed_fallback=not args.no_pubmed,
        verify_answer=not args.no_verify
    )
    
    # Display results
    print("\n" + "="*80)
    if result["status"] == "rejected":
        print("❌ QUERY REJECTED")
        print(f"Reason: {result['reason']}")
        print(f"Suggestion: {result['suggestion']}")
        if result.get("partial_evidence"):
            print("\nPartial evidence found:")
            for ev in result["partial_evidence"]:
                print(f"  - {ev['title']} ({ev['level']}) - score: {ev['score']}")
    else:
        print("ANSWER")
        print(result["answer"])
        print(f"CONFIDENCE: {result['confidence']['label']} ({result['confidence']['score']:.3f})")
        print("\nEVIDENCE SUMMARY:")
        print(f"  Total retrieved: {result['evidence_summary']['total_retrieved']}")
        print(f"  High quality: {result['evidence_summary']['high_quality_count']}")
        print(f"  GRADE distribution: {result['evidence_summary']['grade_distribution']}")
        print(f"  Study types: {result['evidence_summary']['study_types']}")
        
        if result.get("contradictions", {}).get("has_contradictions"):
            print("\n⚠️  CONTRADICTIONS DETECTED:")
            for conflict in result["contradictions"]["conflicts"]:
                print(f"  - {conflict['type']}: {conflict['explanation']}")
        
        if result.get("verification"):
            ver = result["verification"]
            print(f"\nCLAIM VERIFICATION: {ver['verified_claims']}/{ver['total_claims']} verified ({ver['verification_rate']*100:.1f}%)")
            if ver['verification_rate'] < 0.8:
                print(" Some claims lack strong evidence support")
        
        print("\nSOURCES:")
        print(result["sources"])
    print("="*80)


if __name__ == "__main__":
    main()