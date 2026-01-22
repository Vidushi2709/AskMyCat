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
  const formatAnswer = (text: string) => {
    if (!text) return ''
    return text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  }

  if (message.role === 'user') {
    return (
      <div className="flex gap-3 justify-end">
        <div className="flex-1 flex justify-end">
          <div className="bg-blue-500 text-white rounded-lg px-4 py-3 inline-block max-w-[80%]">
            <p className="text-sm">{message.content}</p>
          </div>
        </div>
        <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4 text-white" />
        </div>
      </div>
    )
  }

  const response = message.response

  if (!response && message.content) {
    return (
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1">
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg px-4 py-3">
            <p className="text-sm text-red-800 dark:text-red-200">{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!response) return null

  const isRejected = response.gate_status.includes('rejected')

  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-700 flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-gray-700 dark:text-gray-300" />
      </div>
      
      <div className="flex-1 space-y-4">
        {response.gates && response.gates.length > 0 && (
          <GateStatus gates={response.gates} status={response.gate_status} />
        )}

        {isRejected ? (
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg p-4">
            <p className="font-semibold text-red-800 dark:text-red-200 mb-2">Query Rejected</p>
            <p className="text-sm text-gray-700 dark:text-gray-300">
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
            {response.answer && (
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-gray-900 dark:text-white">Answer</h3>
                  <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                    <span>⏱️ {response.response_time_ms.toFixed(0)}ms</span>
                    {response.cache_hit && <span>💾 Cached</span>}
                  </div>
                </div>
                <div 
                  className="text-sm whitespace-pre-wrap leading-relaxed text-gray-800 dark:text-gray-200" 
                  dangerouslySetInnerHTML={{ __html: formatAnswer(response.answer) }}
                />
                
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    response.confidence === 'high' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    response.confidence === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                    'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {response.confidence === 'high' ? 'HIGHEST' : response.confidence.toUpperCase()} Confidence
                  </span>
                </div>

                {response.follow_up_questions && response.follow_up_questions.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">If you want, I can also:</p>
                    <ul className="space-y-1.5">
                      {response.follow_up_questions.map((question, idx) => (
                        <li key={idx}>
                          <button
                            onClick={() => onFollowUp(question)}
                            className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:underline text-left"
                          >
                            {idx + 1}. {question}
                          </button>
                        </li>
                      ))}
                    </ul>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 italic">Which one would you like me to answer?</p>
                  </div>
                )}
              </div>
            )}

            {response.has_contradictions && response.contradictions && (
              <ContradictionAlert contradictions={response.contradictions} />
            )}

            {response.evidence_chain && (
              <EvidenceVerification evidenceChain={response.evidence_chain} />
            )}

            {response.evidence_passages.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  📚 Evidence ({response.evidence_count} passages)
                </h4>
                <div className="space-y-2">
                  {response.evidence_passages.slice(0, 3).map((evidence, idx) => (
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
