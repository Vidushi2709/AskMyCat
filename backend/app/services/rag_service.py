import sys
from pathlib import Path
import time
import logging
from typing import Optional

# Add parent directory to import retreiver
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from retreiver import RetrievalPipeline

logger = logging.getLogger(__name__)

class RAGService:
    """Wrapper service for RetrievalPipeline with lifecycle management"""
    
    def __init__(self, device: str = "cuda", enable_cache: bool = True):
        self.device = device
        self.enable_cache = enable_cache
        self.pipeline: Optional[RetrievalPipeline] = None
        self.start_time = time.time()
        
    def initialize(self):
        """Initialize the RAG pipeline"""
        try:
            logger.info(f"Initializing RAG pipeline on device: {self.device}")
            self.pipeline = RetrievalPipeline(
                device=self.device,
                enable_cache=self.enable_cache
            )
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        enable_gates: bool = True,
        verify_evidence: bool = False,
        detect_conflicts: bool = True,
        use_llm: bool = True
    ) -> dict:
        """Execute RAG query"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        start_time = time.time()
        
        try:
            result = self.pipeline.answer_query(
                user_query=query,
                top_k=top_k,
                threshold=threshold,
                use_llm=use_llm,
                enable_gates=enable_gates,
                verify_chain=verify_evidence,
                detect_conflicts=detect_conflicts
            )
            
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
        threshold: float = 0.5,
        num_diagnoses: int = 5
    ) -> dict:
        """Generate differential diagnosis"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        start_time = time.time()
        
        try:
            result = self.pipeline.generate_differential_diagnosis(
                user_query=query,
                top_k=top_k,
                threshold=threshold,
                num_diagnoses=num_diagnoses
            )
            
            result["response_time_ms"] = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"DDX generation failed: {e}")
            raise
    
    def _format_response(self, result: dict) -> dict:
        """Format pipeline result for API response"""
        # Extract gate results
        gates = []
        if "gate_results" in result:
            for gate_name, gate_data in result["gate_results"].items():
                gates.append({
                    "gate_name": gate_name,
                    "passed": gate_data.get("passed", False),
                    "score": gate_data.get("score", 0.0),
                    "threshold": gate_data.get("threshold", 0.0),
                    "reason": gate_data.get("reason")
                })
        
        # Format evidence passages
        evidence_passages = []
        for passage_tuple in result.get("filtered_passages", []):
            # Handle tuple format: (text, metadata, similarity)
            if isinstance(passage_tuple, tuple) and len(passage_tuple) >= 3:
                text, metadata, similarity = passage_tuple[0], passage_tuple[1], passage_tuple[2]
            else:
                # Fallback if it's a dict (shouldn't happen, but safe)
                text = passage_tuple.get("text", "") if hasattr(passage_tuple, 'get') else str(passage_tuple)
                metadata = passage_tuple.get("metadata", {}) if hasattr(passage_tuple, 'get') else {}
                similarity = passage_tuple.get("similarity", 0.0) if hasattr(passage_tuple, 'get') else 0.0
            
            energy = 1 - similarity
            
            # Determine confidence level
            if energy < 0.3:
                confidence_level = "HIGH"
            elif energy < 0.5:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
            
            evidence_passages.append({
                "text": text,
                "metadata": metadata,
                "similarity": similarity,
                "energy": energy,
                "confidence_level": confidence_level
            })
        
        # Format contradictions
        contradictions = []
        if result.get("contradictions", {}).get("has_contradictions"):
            for conflict in result["contradictions"].get("conflicts", []):
                contradictions.append({
                    "passage1_idx": conflict.get("passage1_idx"),
                    "passage2_idx": conflict.get("passage2_idx"),
                    "severity": conflict.get("severity"),
                    "explanation": conflict.get("explanation")
                })
        
        # Build formatted response
        formatted = {
            "query": result.get("query", ""),
            "gate_status": result.get("gate_status", "unknown"),
            "gates": gates,
            "answer": result.get("answer"),
            "follow_up_questions": result.get("follow_up_questions", []),
            "confidence": result.get("confidence", "unknown"),
            "evidence_count": len(evidence_passages),
            "evidence_passages": evidence_passages,
            "evidence_chain": result.get("evidence_chain"),
            "verification_rate": result.get("evidence_chain", {}).get("verification_rate"),
            "contradictions": contradictions,
            "has_contradictions": bool(contradictions),
            "response_time_ms": result.get("response_time_ms", 0),
            "cache_hit": result.get("cache_hit", False),
            "warnings": result.get("warnings", [])
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
