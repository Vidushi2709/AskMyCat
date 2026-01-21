from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
import logging
from typing import Any

from app.models.schemas import (
    QueryRequest, QueryResponse,
    DDXRequest, DDXResponse,
    HealthResponse, ErrorResponse
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# Router
router = APIRouter()

# Global service instance (initialized in main.py)
rag_service: RAGService = None

def get_rag_service():
    """Dependency to get RAG service"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return rag_service

@router.post("/query", response_model=QueryResponse, responses={
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def query_rag(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    Main RAG query endpoint
    
    - **query**: Medical question (10-500 chars)
    - **top_k**: Number of passages to retrieve (1-50)
    - **threshold**: Similarity threshold (0.0-1.0)
    - **enable_gates**: Enable multi-level energy gates
    - **verify_evidence**: Verify answer with evidence chain
    - **detect_conflicts**: Detect contradictions in evidence
    - **use_llm**: Generate LLM answer
    """
    try:
        result = service.query(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            enable_gates=request.enable_gates,
            verify_evidence=request.verify_evidence,
            detect_conflicts=request.detect_conflicts,
            use_llm=request.use_llm
        )
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Query execution failed",
                detail=str(e),
                suggestion="Check query format and try again"
            ).dict()
        )

@router.post("/ddx", response_model=DDXResponse, responses={
    500: {"model": ErrorResponse}
})
async def differential_diagnosis(
    request: DDXRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    Differential diagnosis endpoint
    
    - **query**: Clinical presentation (symptoms, findings)
    - **top_k**: Number of passages to retrieve
    - **threshold**: Similarity threshold
    - **num_diagnoses**: Number of top diagnoses to return (1-20)
    """
    try:
        result = service.differential_diagnosis(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            num_diagnoses=request.num_diagnoses
        )
        return DDXResponse(**result)
        
    except Exception as e:
        logger.error(f"DDX failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check(service: RAGService = Depends(get_rag_service)):
    """
    Health check endpoint
    
    Returns service status, device info, and cache statistics
    """
    return HealthResponse(**service.get_health())

@router.post("/cache/clear")
async def clear_cache(
    background_tasks: BackgroundTasks,
    service: RAGService = Depends(get_rag_service)
):
    """
    Clear cache endpoint (admin only in production)
    
    Clears both retrieval and LLM caches
    """
    background_tasks.add_task(service.clear_cache)
    return {"message": "Cache clear scheduled"}

@router.get("/cache/stats")
async def cache_stats(service: RAGService = Depends(get_rag_service)):
    """
    Get cache statistics
    
    Returns hit rates, cache size, and performance metrics
    """
    health = service.get_health()
    stats = health["cache_stats"]
    
    total_requests = sum(stats.values())
    hit_rate = (
        (stats.get("retrieval_hits", 0) + stats.get("llm_hits", 0)) / max(1, total_requests)
    )
    
    return {
        "cache_size": len(service.pipeline.cache) if service.pipeline.cache else 0,
        "stats": stats,
        "hit_rate": f"{hit_rate:.2%}",
        "total_requests": total_requests
    }

# Stream endpoint (optional - for real-time responses)
@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    Streaming query endpoint (Server-Sent Events)
    
    Streams gate results and answer generation in real-time
    """
    async def event_generator():
        import json
        
        # This is a simplified version - you'd need to modify your pipeline
        # to yield intermediate results
        
        yield f"data: {json.dumps({'event': 'started', 'query': request.query})}\n\n"
        
        try:
            result = service.query(
                query=request.query,
                top_k=request.top_k,
                threshold=request.threshold,
                enable_gates=request.enable_gates,
                verify_evidence=request.verify_evidence,
                detect_conflicts=request.detect_conflicts,
                use_llm=request.use_llm
            )
            
            # Stream gates
            for gate in result.get("gates", []):
                yield f"data: {json.dumps({'event': 'gate', 'data': gate})}\n\n"
            
            # Stream answer
            yield f"data: {json.dumps({'event': 'answer', 'data': result.get('answer')})}\n\n"
            
            # Stream complete
            yield f"data: {json.dumps({'event': 'complete', 'data': result})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
