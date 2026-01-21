import axios from 'axios'
import { QuerySettings, QueryResponse } from '@/types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export async function queryRAG(query: string, settings: QuerySettings): Promise<QueryResponse> {
  const response = await api.post('/query', {
    query,
    ...settings,
  })
  return response.data
}

export async function getHealth() {
  const response = await api.get('/health')
  return response.data
}

export async function getCacheStats() {
  const response = await api.get('/cache/stats')
  return response.data
}

export async function clearCache() {
  const response = await api.post('/cache/clear')
  return response.data
}

export default api
