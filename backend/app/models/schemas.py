from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500, description="Medical question to ask")
    top_k: int = Field(10, ge=1, le=50, description="Number of passages to retrieve")
    min_evidence_level: str = Field("LOW", description="Minimum GRADE evidence level (HIGH, MODERATE, LOW, VERY_LOW)")
    use_pubmed_fallback: bool = Field(True, description="Enable PubMed search fallback")
    verify_answer: bool = Field(True, description="Verify answer with BiomedNLI")
    
    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @field_validator('min_evidence_level')
    def validate_evidence_level(cls, v):
        valid_levels = ["HIGH", "MODERATE", "LOW", "VERY_LOW"]
        if v not in valid_levels:
            raise ValueError(f"Evidence level must be one of {valid_levels}")
        return v

class DDXRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    num_diagnoses: int = Field(5, ge=1, le=20, description="Number of differential diagnoses to return")

# Response Models
class EvidencePassage(BaseModel):
    index: int
    text: str
    metadata: Dict[str, Any]
    evidence_level: str  # "HIGH", "MODERATE", "LOW", "VERY_LOW", "UNKNOWN"
    study_design: str  # Study design type

class Contradiction(BaseModel):
    type: str  # "statistical", "directional"
    source_1: str
    source_2: str
    severity: str  # "high", "moderate", "minor"
    explanation: str

class VerificationClaim(BaseModel):
    claim: str
    verified: bool
    confidence: float
    supporting_evidence: Optional[str] = None
    evidence_quality: Optional[str] = None

class ClaimVerification(BaseModel):
    verified: bool
    total_claims: int
    verified_claims: int
    verification_rate: float
    unverified_count: int
    claims: List[VerificationClaim] = []

class ConfidenceScore(BaseModel):
    label: str  # "HIGH", "MODERATE", "LOW", "VERY_LOW"
    score: float  # 0.0-1.0
    components: Optional[Dict[str, float]] = None

class EvidenceSummary(BaseModel):
    total_retrieved: int
    high_quality_count: int
    grade_distribution: Dict[str, int]
    study_types: Dict[str, int]

class QueryResponse(BaseModel):
    query: str
    status: str  # "success", "rejected"
    answer: Optional[str] = None
    confidence: Optional[ConfidenceScore] = None
    
    # Evidence
    evidence_summary: Optional[EvidenceSummary] = None
    evidence_count: int = 0
    evidence_passages: List[EvidencePassage] = []
    sources: str = ""
    
    # Verification
    verification: Optional[ClaimVerification] = None
    
    # Contradictions
    contradictions: Optional[List[Contradiction]] = None
    has_contradictions: bool = False
    
    # Metadata
    response_time_ms: Optional[float] = None
    reason: Optional[str] = None
    suggestion: Optional[str] = None

class DDXResponse(BaseModel):
    query: str
    status: str
    answer: Optional[str] = None
    confidence: Optional[ConfidenceScore] = None
    evidence_summary: EvidenceSummary
    evidence_count: int
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
