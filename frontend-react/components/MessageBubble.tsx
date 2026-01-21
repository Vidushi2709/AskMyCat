import { ChatMessage } from '@/types'
import { User, Bot, AlertTriangle } from 'lucide-react'
import GateStatus from './GateStatus'
import EvidenceCard from './EvidenceCard'
import ContradictionAlert from './ContradictionAlert'
import EvidenceVerification from './EvidenceVerification'

interface MessageBubbleProps {
  message: ChatMessage
  onFollowUp: (question: string) => void
}

export default function MessageBubble({ message, onFollowUp }: MessageBubbleProps) {
  if (message.role === 'user') {
    return (
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4 text-primary-foreground" />
        </div>
        <div className="flex-1">
          <div className="bg-primary/10 rounded-lg px-4 py-3 inline-block max-w-[80%]">
            <p className="text-sm">{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  const response = message.response

  // Error message
  if (!response && message.content) {
    return (
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-destructive flex items-center justify-center flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-destructive-foreground" />
        </div>
        <div className="flex-1">
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg px-4 py-3">
            <p className="text-sm text-destructive">{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!response) return null

  const isRejected = response.gate_status.includes('rejected')

  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-foreground" />
      </div>
      
      <div className="flex-1 space-y-4">
        {/* Gate Status */}
        {response.gates && response.gates.length > 0 && (
          <GateStatus gates={response.gates} status={response.gate_status} />
        )}

        {isRejected ? (
          /* Rejection Message */
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
            <p className="font-semibold text-destructive mb-2">Query Rejected</p>
            <p className="text-sm text-muted-foreground">
              The system rejected this query due to quality concerns. 
              {response.gates.map(g => !g.passed && (
                <span key={g.gate_name} className="block mt-2">
                  • {g.gate_name}: {g.reason}
                </span>
              ))}
            </p>
          </div>
        ) : (
          <>
            {/* Answer */}
            {response.answer && (
              <div className="bg-card border rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Answer</h3>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>⏱️ {response.response_time_ms.toFixed(0)}ms</span>
                    {response.cache_hit && <span>💾 Cached</span>}
                  </div>
                </div>
                <p className="text-sm whitespace-pre-wrap leading-relaxed">{response.answer}</p>
                
                {/* Confidence Badge */}
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    response.confidence === 'high' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    response.confidence === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                    'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {response.confidence.toUpperCase()} Confidence
                  </span>
                </div>
              </div>
            )}

            {/* Follow-up Questions */}
            {response.follow_up_questions && response.follow_up_questions.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-muted-foreground">💡 Suggested Follow-up Questions</h4>
                <div className="flex flex-wrap gap-2">
                  {response.follow_up_questions.map((question, idx) => (
                    <button
                      key={idx}
                      onClick={() => onFollowUp(question)}
                      className="text-xs px-3 py-2 rounded-md bg-secondary hover:bg-secondary/80 text-left transition-colors"
                    >
                      ❓ {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Contradictions */}
            {response.has_contradictions && response.contradictions && (
              <ContradictionAlert contradictions={response.contradictions} />
            )}

            {/* Evidence Verification */}
            {response.evidence_chain && (
              <EvidenceVerification evidenceChain={response.evidence_chain} />
            )}

            {/* Evidence Passages */}
            {response.evidence_passages.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-muted-foreground">
                  📚 Evidence ({response.evidence_count} passages)
                </h4>
                <div className="space-y-2">
                  {response.evidence_passages.map((evidence, idx) => (
                    <EvidenceCard key={idx} evidence={evidence} index={idx + 1} />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
