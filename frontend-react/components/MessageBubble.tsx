import { ChatMessage } from '@/types'
import { User, Bot, AlertTriangle } from 'lucide-react'
import GateStatus from './GateStatus'
import EvidenceCard from './EvidenceCard'
import EvidenceVerification from './EvidenceVerification'

interface MessageBubbleProps {
  message: ChatMessage
  onFollowUp: (question: string) => void
}

export default function MessageBubble({ message, onFollowUp }: MessageBubbleProps) {
  // Helper function to format bold text
  // Format answer and make [Evidence N] citations clickable
  const formatAnswer = (text: string) => {
    if (!text) return ''
    // Bold formatting
    let html = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Evidence citation clickable links
    html = html.replace(/\[Evidence (\d+)\]/g, (match, p1) => {
      return `<a href="#evidence-${p1}" class="text-primary underline hover:opacity-80">[Evidence ${p1}]</a>`
    })
    return html
  }

  if (message.role === 'user') {
    return (
      <div className="flex gap-3 justify-end">
        <div className="flex-1 flex justify-end">
          <div className="bg-primary text-primary-foreground rounded-lg px-4 py-3 inline-block max-w-[80%]">
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
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-destructive flex items-center justify-center flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-destructive-foreground" />
        </div>
        <div className="flex-1">
          <div className="bg-destructive/10 border border-destructive/30 rounded-lg px-4 py-3">
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
      <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-muted-foreground" />
      </div>
      
      <div className="flex-1 space-y-4">
        {/* Gate Status */}
        {response.gates && response.gates.length > 0 && (
          <GateStatus gates={response.gates} status={response.gate_status} />
        )}

        {isRejected ? (
          <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-4">
            <p className="font-semibold text-destructive mb-2">Query Rejected</p>
            <p className="text-sm text-foreground">
              The system rejected this query due to quality concerns.
            </p>
            {response.gates && response.gates.length > 0 && (
              <div className="mt-2">
                {response.gates.filter(g => !g.passed).map(g => (
                  <span key={g.gate_name} className="block mt-1 text-sm">
                    • {g.gate_name}: {g.reason}
                  </span>
                ))}
              </div>
            )}
          </div>
        ) : (
          <>
            {/* Answer */}
            {response.answer && (
              <div className="bg-card border border-border rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-card-foreground">Answer</h3>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>⏱️ {response.response_time_ms.toFixed(0)}ms</span>
                    {response.cache_hit && <span>💾 Cached</span>}
                  </div>
                </div>
                <div 
                  className="text-sm whitespace-pre-wrap leading-relaxed text-foreground" 
                  dangerouslySetInnerHTML={{ __html: formatAnswer(response.answer) }}
                />
                
                {/* Confidence Badge */}
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    response.confidence === 'high' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    response.confidence === 'highest' ? 'bg-green-200 text-green-900 dark:bg-green-800 dark:text-green-100' :
                    response.confidence === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                    response.confidence === 'low' ? 'bg-destructive/10 text-destructive' :
                    'bg-muted/50 text-muted-foreground'
                  }`}>
                    {response.confidence ? response.confidence.toUpperCase() : 'UNKNOWN'} Confidence
                  </span>
                </div>

              </div>
            )}

            {response.verification && (
              <EvidenceVerification verification={response.verification} />
            )}

            {response.evidence_passages && response.evidence_passages.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  📚 Evidence ({response.evidence_count} passages)
                </h4>
                <div className="space-y-2">
                  {response.evidence_passages.slice(0, 3).map((evidence, idx) => (
                    <div key={idx} id={`evidence-${idx + 1}`}>
                      <EvidenceCard evidence={evidence} index={idx + 1} />
                    </div>
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
