// Query Settings with GRADE Evidence Levels
export type EvidenceLevel = 'HIGH' | 'MODERATE' | 'LOW' | 'VERY_LOW'
export type StudyDesign = 'META_ANALYSIS' | 'SYSTEMATIC_REVIEW' | 'RCT' | 'COHORT' | 'CASE_CONTROL' | 'CROSS_SECTIONAL' | 'CASE_SERIES' | 'CASE_REPORT' | 'EXPERT_OPINION' | 'UNKNOWN'

export interface QuerySettings {
  top_k: number
  min_evidence_level: EvidenceLevel
  use_pubmed_fallback: boolean
  verify_answer: boolean
}

// Confidence Score Model
export interface ConfidenceScore {
  label: 'HIGH' | 'MODERATE' | 'LOW' | 'VERY_LOW'
  score: number // 0.0-1.0
  components: {
    evidence_quality: number
    grade_bonus: number
    verification_penalty: number
  }
}

// Evidence Summary
export interface EvidenceSummary {
  total_retrieved: number
  high_quality_count: number
  grade_distribution: Record<string, number>
  study_types: Record<string, number>
}

// Evidence Passage with GRADE Level and Study Design
export interface EvidencePassage {
  index: number
  text: string
  metadata: Record<string, any>
  evidence_level: EvidenceLevel
  study_design: StudyDesign
}

// Verification Claim
export interface VerificationClaim {
  claim: string
  verified: boolean
  confidence: number
  supporting_evidence?: string
  evidence_quality?: EvidenceLevel
}

// Claim Verification
export interface ClaimVerification {
  verified: boolean
  total_claims: number
  verified_claims: number
  verification_rate: number
  unverified_count: number
  claims: VerificationClaim[]
}

// Contradiction Detection
export interface Contradiction {
  type: 'statistical' | 'directional'
  source_1: string
  source_2: string
  severity: 'high' | 'moderate' | 'minor'
  explanation: string
}

// Main Query Response
export interface QueryResponse {
  query: string
  status: 'success' | 'rejected'
  answer?: string
  confidence?: ConfidenceScore
  evidence_summary: EvidenceSummary
  evidence_count: number
  evidence_passages: EvidencePassage[]
  sources: string
  verification?: ClaimVerification
  contradictions: Contradiction[]
  has_contradictions: boolean
  response_time_ms: number
  reason?: string // For rejected queries
  suggestion?: string // For rejected queries
}

// DDX Response
export interface DDXResponse {
  query: string
  status: 'success' | 'rejected'
  answer?: string
  confidence?: ConfidenceScore
  evidence_summary: EvidenceSummary
  evidence_count: number
  response_time_ms: number
}

// Health Check Response
export interface HealthResponse {
  status: 'healthy' | 'unhealthy'
  device: string
  model_loaded: boolean
  cache_stats: Record<string, any>
  uptime_seconds: number
}

// Chat Message
export interface ChatMessage {
  role: 'user' | 'assistant'
  content?: string
  response?: QueryResponse
  timestamp: number
}
