import { AlertTriangle } from 'lucide-react'
import { Contradiction } from '@/types'

interface ContradictionAlertProps {
  contradictions: Contradiction[]
}

export default function ContradictionAlert({ contradictions }: ContradictionAlertProps) {
  return (
    <div className="bg-yellow-50 dark:bg-yellow-950 border-2 border-yellow-300 dark:border-yellow-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle className="w-5 h-5 text-yellow-600" />
        <span className="font-semibold text-yellow-900 dark:text-yellow-100">
          ⚠️ Conflicting Evidence Detected
        </span>
      </div>
      
      <div className="space-y-3">
        {contradictions.map((contradiction, idx) => {
          const severityColor = 
            contradiction.severity === 'critical' ? 'text-red-600 dark:text-red-400' :
            contradiction.severity === 'moderate' ? 'text-yellow-600 dark:text-yellow-400' :
            'text-green-600 dark:text-green-400'
          
          const severityIcon = 
            contradiction.severity === 'critical' ? '🔴' :
            contradiction.severity === 'moderate' ? '🟡' : '🟢'

          return (
            <div key={idx} className="bg-white dark:bg-gray-900 rounded p-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm">{severityIcon}</span>
                <span className={`text-xs font-semibold uppercase ${severityColor}`}>
                  {contradiction.severity} Conflict
                </span>
              </div>
              <p className="text-sm text-muted-foreground">
                Evidence #{contradiction.passage1_idx + 1} conflicts with Evidence #{contradiction.passage2_idx + 1}
              </p>
              <p className="text-sm mt-2">{contradiction.explanation}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
