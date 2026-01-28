import axios from 'axios'
import { QuerySettings, QueryResponse, DDXResponse, HealthResponse, EvidenceLevel } from '@/types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Error handling
api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error)
    throw error
  }
)

/**
 * Query the medical RAG system with GRADE evidence grading
 * @param query - Medical question to ask
 * @param settings - Query settings (top_k, evidence_level, etc.)
 * @returns QueryResponse with answer and evidence
 */
export async function queryRAG(query: string, settings?: Partial<QuerySettings>): Promise<QueryResponse> {
  const defaultSettings: QuerySettings = {
    top_k: 10,
    min_evidence_level: 'MODERATE',
    use_pubmed_fallback: true,
    verify_answer: true,
  }

  const response = await api.post('/query', {
    query,
    ...defaultSettings,
    ...settings,
  })
  return response.data
}

/**
 * Get differential diagnoses for clinical presentation
 * @param query - Clinical presentation/symptoms
 * @param topK - Number of passages to retrieve
 * @param numDiagnoses - Number of diagnoses to return
 * @returns DDXResponse with differential diagnoses
 */
export async function getDifferentialDiagnosis(
  query: string,
  topK: number = 15,
  numDiagnoses: number = 5
): Promise<DDXResponse> {
  const response = await api.post('/ddx', {
    query,
    top_k: topK,
    num_diagnoses: numDiagnoses,
  })
  return response.data
}

/**
 * Get service health status
 * @returns HealthResponse with service status and cache stats
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await api.get('/health')
  return response.data
}

/**
 * Get cache statistics
 * @returns Cache statistics including hit rate and size
 */
export async function getCacheStats() {
  const response = await api.get('/cache/stats')
  return response.data
}

/**
 * Clear the cache
 * @returns Confirmation message
 */
export async function clearCache() {
  const response = await api.post('/cache/clear')
  return response.data
}

/**
 * Get service info
 * @returns Service name, version, and API docs URLs
 */
export async function getServiceInfo() {
  const response = await api.get('/')
  return response.data
}

// Helper functions for common queries

/**
 * Query with HIGH evidence level (most strict)
 */
export async function queryHighEvidence(query: string): Promise<QueryResponse> {
  return queryRAG(query, { min_evidence_level: 'HIGH' })
}

/**
 * Query with MODERATE evidence level (balanced)
 */
export async function queryModerateEvidence(query: string): Promise<QueryResponse> {
  return queryRAG(query, { min_evidence_level: 'MODERATE' })
}

/**
 * Query with LOW evidence level (permissive)
 */
export async function queryLowEvidence(query: string): Promise<QueryResponse> {
  return queryRAG(query, { min_evidence_level: 'LOW' })
}

/**
 * Query without verification (faster)
 */
export async function queryFast(query: string): Promise<QueryResponse> {
  return queryRAG(query, { verify_answer: false })
}

/**
 * Query with maximum evidence (50 passages)
 */
export async function queryComprehensive(query: string): Promise<QueryResponse> {
  return queryRAG(query, { top_k: 50, verify_answer: true })
}

export default api
