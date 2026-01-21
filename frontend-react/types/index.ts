export interface QuerySettings {
  top_k: number
  threshold: number
  enable_gates: boolean
  verify_evidence: boolean
  detect_conflicts: boolean
  use_llm: boolean
}

export interface GateResult {
  gate_name: string
  passed: boolean
  score: number
  threshold: number
  reason?: string
}

export interface EvidencePassage {
  text: string
  metadata: Record<string, any>
  similarity: number
  energy: number
  confidence_level: string
}

export interface Contradiction {
  passage1_idx: number
  passage2_idx: number
  severity: string
  explanation: string
}

export interface EvidenceChain {
  verification_rate: number
  verified_sentences: number
  unverified_sentences: number
  sentences: Array<{
    sentence: string
    verified: boolean
    confidence: number
    citations: number[]
  }>
}

export interface QueryResponse {
  query: string
  gate_status: string
  gates: GateResult[]
  answer?: string
  follow_up_questions?: string[]
  confidence: string
  evidence_count: number
  evidence_passages: EvidencePassage[]
  evidence_chain?: EvidenceChain
  verification_rate?: number
  contradictions?: Contradiction[]
  has_contradictions: boolean
  response_time_ms: number
  cache_hit: boolean
  warnings: string[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content?: string
  response?: QueryResponse
  timestamp: number
}
