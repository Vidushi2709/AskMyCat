import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { EvidencePassage } from '@/types'

interface EvidenceCardProps {
  evidence: EvidencePassage
  index: number
}

export default function EvidenceCard({ evidence, index }: EvidenceCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const confidenceBadge = evidence.confidence_level === 'HIGH' ? '🟢' : 
                         evidence.confidence_level === 'MEDIUM' ? '🟡' : '🔴'

  return (
    <div className="border rounded-md overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between bg-secondary hover:bg-secondary/80 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">📄 Evidence {index}</span>
          <span className="text-xs">{confidenceBadge} {evidence.confidence_level}</span>
          <span className="text-xs text-muted-foreground">
            Score: {evidence.similarity.toFixed(3)}
          </span>
        </div>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      
      {isExpanded && (
        <div className="p-4 bg-card space-y-3">
          <p className="text-sm leading-relaxed">{evidence.text}</p>
          
          {Object.keys(evidence.metadata).length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                Metadata
              </summary>
              <pre className="mt-2 p-2 bg-secondary rounded text-xs overflow-x-auto">
                {JSON.stringify(evidence.metadata, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  )
}
