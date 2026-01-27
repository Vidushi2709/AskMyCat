import { GateResult } from '@/types'
import { CheckCircle, XCircle, Shield } from 'lucide-react'

interface GateStatusProps {
  gates: GateResult[]
  status: string
}

export default function GateStatus({ gates, status }: GateStatusProps) {
  return (
    <div className="bg-card border rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <Shield className="w-4 h-4" />
        <span className="font-semibold text-sm">Energy Gates</span>
        <span className={`text-xs px-2 py-0.5 rounded-full ml-auto ${
          status.includes('passed') 
            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
            : 'bg-destructive/10 text-destructive'
        }`}>
          {status.toUpperCase()}
        </span>
      </div>
      
      <div className="grid grid-cols-3 gap-3">
        {gates.map((gate) => (
          <div
            key={gate.gate_name}
            className={`p-3 rounded-md border-2 ${
              gate.passed
                ? 'bg-green-50 border-green-300 dark:bg-green-950 dark:border-green-800'
                : 'bg-destructive/5 border-destructive/30'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium">
                {gate.gate_name.replace('gate_', 'Gate ').replace('_', ' ')}
              </span>
              {gate.passed ? (
                <CheckCircle className="w-4 h-4 text-green-600" />
              ) : (
                <XCircle className="w-4 h-4 text-destructive" />
              )}
            </div>
            <div className="text-xs text-muted-foreground">
              Score: {(gate.score * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-muted-foreground">
              Threshold: {(gate.threshold * 100).toFixed(0)}%
            </div>
            {gate.reason && (
              <div className="text-xs text-muted-foreground mt-1 line-clamp-2">
                {gate.reason}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
