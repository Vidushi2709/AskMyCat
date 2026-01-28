'use client'

import { ChatMessage } from '@/types'
import { User, Bot, AlertTriangle, CheckCircle, AlertCircle, TrendingUp } from 'lucide-react'
import EvidenceCard from './EvidenceCard'
import EvidenceVerification from './EvidenceVerification'
import ContradictionAlert from './ContradictionAlert'

interface MessageBubbleProps {
  message: ChatMessage
  onFollowUp: (question: string) => void
}

export default function MessageBubbleGRADE({ message, onFollowUp }: MessageBubbleProps) {
  if (message.role === 'user') {
    return (
      <div className="flex gap-3 justify-end mb-4">
        <div className="flex-1 flex justify-end">
          <div className="bg-primary text-primary-foreground rounded-lg px-4 py-3 inline-block max-w-[80%] shadow-sm">
            <p className="text-sm">{message.content}</p>
          </div>
        </div>
        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4" />
        </div>
      </div>
    )
  }

  const response = message.response

  // Error message
  if (!response && message.content) {
    return (
      <div className="flex gap-3 mb-4">
        <div className="w-8 h-8 rounded-full bg-destructive flex items-center justify-center flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-destructive-foreground" />
        </div>
        <div className="flex-1">
          <div className="bg-destructive/10 border border-destructive/30 rounded-lg px-4 py-3 shadow-sm">
            <p className="text-sm text-destructive font-medium">{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!response) return null

  const isRejected = response.status === 'rejected'

  // Confidence color mapping
  const getConfidenceColor = (label?: string) => {
    switch (label) {
      case 'HIGH':
        return 'bg-green-50 border-green-200 text-green-900'
      case 'MODERATE':
        return 'bg-blue-50 border-blue-200 text-blue-900'
      case 'LOW':
        return 'bg-yellow-50 border-yellow-200 text-yellow-900'
      case 'VERY_LOW':
        return 'bg-red-50 border-red-200 text-red-900'
      default:
        return 'bg-gray-50 border-gray-200 text-gray-900'
    }
  }

  const getConfidenceIcon = (label?: string) => {
    switch (label) {
      case 'HIGH':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'MODERATE':
        return <TrendingUp className="w-4 h-4 text-blue-600" />
      case 'LOW':
        return <AlertCircle className="w-4 h-4 text-yellow-600" />
      case 'VERY_LOW':
        return <AlertTriangle className="w-4 h-4 text-red-600" />
      default:
        return <AlertCircle className="w-4 h-4 text-gray-600" />
    }
  }

  return (
    <div className="flex gap-3 mb-4">
      <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-muted-foreground" />
      </div>

      <div className="flex-1 space-y-4">
        {isRejected ? (
          <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-semibold text-destructive">Query Could Not Be Answered</p>
                <p className="text-sm text-foreground mt-1">{response.reason}</p>
                {response.suggestion && (
                  <p className="text-sm text-muted-foreground mt-2">
                    💡 Suggestion: {response.suggestion}
                  </p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Confidence Badge - Compact */}
            {response.confidence && response.confidence.score !== null && (
              <div className={`border rounded-lg p-3 ${getConfidenceColor(response.confidence.label)}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {getConfidenceIcon(response.confidence.label)}
                    <div>
                      <p className="font-semibold text-sm">{response.confidence.label} Confidence</p>
                    </div>
                  </div>
                  <span className="text-xs opacity-60">Score: {(response.confidence.score * 100).toFixed(1)}%</span>
                </div>
              </div>
            )}

            {/* Answer - Main Content */}
            {response.answer && (
              <div className="bg-card border border-border rounded-lg p-4 space-y-3 shadow-sm">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <p className="text-sm text-foreground whitespace-pre-wrap">{response.answer}</p>
                </div>
              </div>
            )}

            {/* Sources - Clean Display */}
            {response.sources && (
              <div className="bg-muted rounded-lg p-4 space-y-2 shadow-sm">
                <h3 className="font-semibold text-sm text-foreground">Sources</h3>
                <pre className="text-xs overflow-x-auto whitespace-pre-wrap text-muted-foreground bg-background rounded p-2">
                  {response.sources}
                </pre>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
