import sys
from pathlib import Path
import time
import logging
from typing import Optional

# Add parent directory to import retreiver
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from retreiverv2 import (
    MedicalRetrievalPipeline, 
    EvidenceLevel, 
    MedicalEvidence,
    StudyDesign
)

logger = logging.getLogger(__name__)

class RAGService:
    """Wrapper service for MedicalRetrievalPipeline with lifecycle management"""
    
    def __init__(self, device: str = "cuda", enable_cache: bool = True):
        self.device = device
        self.enable_cache = enable_cache
        self.pipeline: Optional[MedicalRetrievalPipeline] = None
        self.start_time = time.time()
        
    def initialize(self):
        """Initialize the medical RAG pipeline"""
        try:
            logger.info(f"Initializing Medical RAG pipeline on device: {self.device}")
            self.pipeline = MedicalRetrievalPipeline(
                device=self.device,
                enable_cache=self.enable_cache
            )
            logger.info("Medical RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Medical RAG pipeline: {e}")
            raise
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        min_evidence_level: str = "LOW",
        use_pubmed_fallback: bool = True,
        verify_answer: bool = True
    ) -> dict:
        """Execute medical RAG query with GRADE evidence grading"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        start_time = time.time()
        
        try:
            # Convert string evidence level to enum
            evidence_level = EvidenceLevel[min_evidence_level]
            
            result = self.pipeline.answer_query(
                query=query,
                top_k=top_k,
                min_evidence_level=evidence_level,
                use_pubmed_fallback=use_pubmed_fallback,
                verify_answer=verify_answer
            )
            
            logger.info(f"Pipeline returned result with status: {result.get('status')}")
            
            # Add response time
            result["response_time_ms"] = (time.time() - start_time) * 1000
            
            # Format for API response
            return self._format_response(result)
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def differential_diagnosis(
        self,
        query: str,
        top_k: int = 10,
        num_diagnoses: int = 5
    ) -> dict:
        """Generate differential diagnosis using evidence retrieval"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        start_time = time.time()
        
        try:
            # Append DDX context to query
            ddx_query = f"What are the differential diagnoses for: {query}"
            
            result = self.pipeline.answer_query(
                query=ddx_query,
                top_k=top_k,
                min_evidence_level=EvidenceLevel.MODERATE,
                use_pubmed_fallback=True,
                verify_answer=True
            )
            
            result["response_time_ms"] = (time.time() - start_time) * 1000
            result["num_diagnoses"] = num_diagnoses
            
            return self._format_response(result)
            
        except Exception as e:
            logger.error(f"DDX generation failed: {e}")
            raise
    
    def _format_response(self, result: dict) -> dict:
        """Format MedicalRetrievalPipeline result for API response - SIMPLIFIED"""
        
        # Handle rejected queries
        if result.get("status") == "rejected":
            return {
                "query": result.get("query", ""),
                "status": "rejected",
                "answer": "Insufficient evidence to provide a confident answer. Please consult primary literature or clinical guidelines.",
                "reason": result.get("reason", "Insufficient or inconsistent evidence"),
                "suggestion": result.get("suggestion", ""),
                "confidence": {
                    "label": "VERY_LOW",
                    "score": 0.0
                },
                "sources": "",
                "evidence_count": 0
            }
        
        # Extract confidence
        confidence_data = result.get("confidence", {})
        confidence = {
            "label": confidence_data.get("label", "UNKNOWN"),
            "score": confidence_data.get("score", 0.0)
        }
        
        # Simple, clean response: just answer + sources
        formatted = {
            "query": result.get("query", ""),
            "status": result.get("status", "success"),
            "answer": result.get("answer", ""),
            "confidence": confidence,
            "sources": result.get("sources", ""),
            "evidence_count": len(result.get("filtered_passages", []))
        }
        
        return formatted
    
    def get_health(self) -> dict:
        """Get service health status"""
        cache_stats = {}
        if self.pipeline and self.pipeline.cache:
            cache_stats = self.pipeline.cache_stats.copy()
        
        return {
            "status": "healthy" if self.pipeline else "unhealthy",
            "device": self.device,
            "model_loaded": self.pipeline is not None,
            "cache_stats": cache_stats,
            "uptime_seconds": time.time() - self.start_time
        }
    
    def clear_cache(self):
        """Clear the cache"""
        if self.pipeline:
            self.pipeline.clear_cache()
    
    def shutdown(self):
        """Cleanup resources"""
        if self.pipeline and self.pipeline.cache:
            self.pipeline.cache.close()
        logger.info("RAG service shutdown complete")
