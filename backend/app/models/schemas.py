from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500, description="Medical question to ask")
    top_k: int = Field(10, ge=1, le=50, description="Number of passages to retrieve")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    enable_gates: bool = Field(True, description="Enable multi-level energy gates")
    verify_evidence: bool = Field(False, description="Verify answer with evidence chain")
    detect_conflicts: bool = Field(True, description="Detect contradictions in evidence")
    use_llm: bool = Field(True, description="Generate LLM answer")
    
    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class DDXRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    num_diagnoses: int = Field(5, ge=1, le=20, description="Number of differential diagnoses to return")

# Response Models
class GateResult(BaseModel):
    gate_name: str
    passed: bool
    score: float
    threshold: float
    reason: Optional[str] = None

class EvidencePassage(BaseModel):
    text: str
    metadata: Dict[str, Any]
    similarity: float
    energy: float
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"

class Contradiction(BaseModel):
    passage1_idx: int
    passage2_idx: int
    severity: str  # "critical", "moderate", "minor"
    explanation: str

class EvidenceVerification(BaseModel):
    sentence: str
    verified: bool
    confidence: float
    citations: List[int]

class QueryResponse(BaseModel):
    query: str
    gate_status: str  # "accepted", "rejected", "bypassed"
    gates: List[GateResult]
    answer: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None
    confidence: str  # "high", "medium", "low", "rejected"
    
    # Evidence
    evidence_count: int
    evidence_passages: List[EvidencePassage]
    
    # Verification
    evidence_chain: Optional[Dict[str, Any]] = None
    verification_rate: Optional[float] = None
    
    # Contradictions
    contradictions: Optional[List[Contradiction]] = None
    has_contradictions: bool = False
    
    # Metadata
    response_time_ms: float
    cache_hit: bool
    warnings: List[str] = []

class DDXResponse(BaseModel):
    query: str
    diagnoses: List[Dict[str, Any]]
    response_time_ms: float

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    cache_stats: Dict[str, Any]
    uptime_seconds: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    suggestion: Optional[str] = None
