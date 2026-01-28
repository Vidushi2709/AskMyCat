'use client'

import { EvidenceLevel, StudyDesign } from '@/types'
import { ChevronDown } from 'lucide-react'
import { useState } from 'react'

interface EvidenceCardProps {
  evidence: {
    text: string
    metadata: Record<string, any>
    evidence_level: EvidenceLevel
    study_design: StudyDesign
  }
  index: number
}

export default function EvidenceCardGRADE({ evidence, index }: EvidenceCardProps) {
  const [isOpen, setIsOpen] = useState(false)

  const getGradeColor = (level: EvidenceLevel) => {
    switch (level) {
      case 'HIGH':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'MODERATE':
        return 'bg-blue-100 text-blue-800 border-blue-300'
      case 'LOW':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'VERY_LOW':
        return 'bg-red-100 text-red-800 border-red-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  const getStudyDesignLabel = (design: StudyDesign) => {
    const labels: Record<StudyDesign, string> = {
      META_ANALYSIS: '📊 Meta-Analysis',
      SYSTEMATIC_REVIEW: '📚 Systematic Review',
      RCT: '🧪 Randomized Controlled Trial',
      COHORT: '👥 Cohort Study',
      CASE_CONTROL: '🔍 Case-Control',
      CROSS_SECTIONAL: '📈 Cross-Sectional',
      CASE_SERIES: '📋 Case Series',
      CASE_REPORT: '📄 Case Report',
      EXPERT_OPINION: '💬 Expert Opinion',
      UNKNOWN: '❓ Unknown Design',
    }
    return labels[design]
  }

  // Truncate text for preview
  const preview = evidence.text.length > 200 
    ? evidence.text.substring(0, 200) + '...' 
    : evidence.text

  return (
    <div
      className={`rounded-lg border transition-all cursor-pointer hover:shadow-md ${
        isOpen 
          ? 'border-primary bg-card shadow-md' 
          : 'border-border bg-card/50 hover:border-border'
      }`}
      onClick={() => setIsOpen(!isOpen)}
    >
      <div className="p-3 flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="text-xs font-bold bg-primary/10 text-primary px-2 py-1 rounded">
              [{index}]
            </span>
            <span className={`text-xs font-semibold px-2 py-1 rounded border ${getGradeColor(evidence.evidence_level)}`}>
              {evidence.evidence_level}
            </span>
            <span className="text-xs bg-muted px-2 py-1 rounded text-muted-foreground">
              {getStudyDesignLabel(evidence.study_design)}
            </span>
          </div>

          {/* Metadata */}
          {(evidence.metadata?.title || evidence.metadata?.year) && (
            <div className="text-xs text-muted-foreground mb-2 line-clamp-1">
              {evidence.metadata.title && (
                <span className="font-medium">{evidence.metadata.title}</span>
              )}
              {evidence.metadata.year && (
                <span> ({evidence.metadata.year})</span>
              )}
            </div>
          )}

          {/* Preview */}
          <p className="text-sm text-foreground line-clamp-2 mb-1">
            {preview}
          </p>
        </div>

        <ChevronDown
          className={`w-4 h-4 text-muted-foreground flex-shrink-0 transition-transform ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </div>

      {/* Full Text */}
      {isOpen && (
        <div className="border-t border-border px-3 py-3 bg-muted/30 text-sm space-y-2">
          <div>
            <p className="text-sm text-foreground whitespace-pre-wrap">{evidence.text}</p>
          </div>

          {/* Additional Metadata */}
          {Object.keys(evidence.metadata).length > 1 && (
            <details className="text-xs text-muted-foreground cursor-pointer pt-2 border-t border-border">
              <summary className="font-semibold">Metadata</summary>
              <div className="mt-2 space-y-1 pl-2">
                {Object.entries(evidence.metadata).map(([key, value]) => (
                  <div key={key}>
                    <span className="font-medium">{key}:</span>{' '}
                    <span>{String(value)}</span>
                  </div>
                ))}
              </div>
            </details>
          )}

          {/* External Link */}
          {evidence.metadata?.url && (
            <a
              href={evidence.metadata.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-primary hover:underline text-xs mt-2"
              onClick={(e) => e.stopPropagation()}
            >
              🔗 View Source
            </a>
          )}
        </div>
      )}
    </div>
  )
}
