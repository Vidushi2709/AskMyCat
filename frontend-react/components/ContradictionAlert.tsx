'use client'

import { AlertTriangle, AlertCircle, HelpCircle } from 'lucide-react'
import { Contradiction } from '@/types'

interface ContradictionAlertProps {
  contradictions?: Contradiction | Contradiction[]
}

export default function ContradictionAlert({ contradictions }: ContradictionAlertProps) {
  if (!contradictions) return null

  // Handle both single contradiction and array
  const items = Array.isArray(contradictions) ? contradictions : [contradictions]
  if (items.length === 0) return null

  const getTypeConfig = (type: string) => {
    switch (type) {
      case 'conflicting_evidence':
        return {
          icon: AlertTriangle,
          bg: 'bg-red-50 dark:bg-red-950',
          border: 'border-red-300 dark:border-red-800',
          textColor: 'text-red-700 dark:text-red-200',
          label: '🔴 Conflicting Evidence',
        }
      case 'insufficient_evidence':
        return {
          icon: HelpCircle,
          bg: 'bg-amber-50 dark:bg-amber-950',
          border: 'border-amber-300 dark:border-amber-800',
          textColor: 'text-amber-700 dark:text-amber-200',
          label: '🟡 Insufficient Evidence',
        }
      case 'unclear_mechanism':
      default:
        return {
          icon: AlertCircle,
          bg: 'bg-blue-50 dark:bg-blue-950',
          border: 'border-blue-300 dark:border-blue-800',
          textColor: 'text-blue-700 dark:text-blue-200',
          label: 'ℹ️ Unclear Mechanism',
        }
    }
  }

  return (
    <div className="space-y-2">
      {items.map((contradiction, idx) => {
        const config = getTypeConfig(contradiction.type)
        const Icon = config.icon

        return (
          <div
            key={idx}
            className={`${config.bg} ${config.border} border-2 rounded-lg p-4`}
          >
            <div className="flex items-start gap-3">
              <Icon className={`w-5 h-5 ${config.textColor} mt-0.5 flex-shrink-0`} />
              <div className="flex-1">
                <p className={`font-semibold ${config.textColor} mb-1`}>
                  {config.label}
                </p>
                {contradiction.description && (
                  <p className="text-sm text-muted-foreground mb-2">
                    {contradiction.description}
                  </p>
                )}
                {contradiction.sources && contradiction.sources.length > 0 && (
                  <div className="text-xs space-y-1">
                    <p className="font-medium text-muted-foreground">Conflicting Sources:</p>
                    <ul className="list-disc list-inside space-y-0.5">
                      {contradiction.sources.map((source, sidx) => (
                        <li key={sidx} className="text-muted-foreground">
                          {source}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
