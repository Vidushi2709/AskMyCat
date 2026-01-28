import { useState } from 'react'
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react'
import { EvidencePassage } from '@/types'

interface EvidenceCardProps {
  evidence: EvidencePassage
  index: number
}

export default function EvidenceCard({ evidence, index }: EvidenceCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const levelBadge = evidence.evidence_level === 'HIGH' ? '🟢' : 
                     evidence.evidence_level === 'MODERATE' ? '🔵' : 
                     evidence.evidence_level === 'LOW' ? '🟡' : '🔴'

  const studyDesignEmoji = evidence.study_design === 'META_ANALYSIS' ? '📊' :
                           evidence.study_design === 'SYSTEMATIC_REVIEW' ? '📚' :
                           evidence.study_design === 'RCT' ? '🧪' :
                           evidence.study_design === 'COHORT' ? '👥' :
                           evidence.study_design === 'CASE_CONTROL' ? '📋' :
                           evidence.study_design === 'CROSS_SECTIONAL' ? '📈' :
                           evidence.study_design === 'CASE_SERIES' ? '📑' :
                           evidence.study_design === 'CASE_REPORT' ? '📄' : '❓'

  return (
    <div className="border rounded-md overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between bg-secondary hover:bg-secondary/80 transition-colors text-left"
      >
        <div className="flex items-center gap-2 flex-1">
          <span className="text-sm font-medium">Evidence {index}</span>
          <span className="text-xs">{levelBadge} {evidence.evidence_level}</span>
          <span className="text-xs">{studyDesignEmoji} {evidence.study_design.replace('_', ' ')}</span>
        </div>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      
      {isExpanded && (
        <div className="p-4 bg-card space-y-3">
          <p className="text-sm leading-relaxed text-foreground">{evidence.text}</p>
          
          {evidence.metadata && Object.keys(evidence.metadata).length > 0 && (
            <div className="text-xs space-y-2">
              <details className="cursor-pointer">
                <summary className="text-muted-foreground hover:text-foreground font-medium">
                  Metadata & Details
                </summary>
                <div className="mt-2 p-2 bg-secondary rounded space-y-1">
                  {evidence.metadata.title && (
                    <p><span className="font-medium">Title:</span> {evidence.metadata.title}</p>
                  )}
                  {evidence.metadata.year && (
                    <p><span className="font-medium">Year:</span> {evidence.metadata.year}</p>
                  )}
                  {evidence.metadata.authors && (
                    <p><span className="font-medium">Authors:</span> {evidence.metadata.authors}</p>
                  )}
                  {evidence.metadata.journal && (
                    <p><span className="font-medium">Journal:</span> {evidence.metadata.journal}</p>
                  )}
                </div>
              </details>
              
              {evidence.metadata.url && (
                <a 
                  href={evidence.metadata.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                >
                  View Source <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
