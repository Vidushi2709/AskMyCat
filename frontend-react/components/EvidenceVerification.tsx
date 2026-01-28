import { CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import { ClaimVerification } from '@/types'

interface EvidenceVerificationProps {
  verification?: ClaimVerification
}

export default function EvidenceVerification({ verification }: EvidenceVerificationProps) {
  if (!verification) {
    return (
      <div className="bg-card border rounded-lg p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <AlertCircle className="w-4 h-4" />
          <p className="text-sm">No verification data available</p>
        </div>
      </div>
    )
  }

  const verificationRate = verification.verification_rate * 100

  return (
    <div className="bg-card border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-sm">Claim Verification</h4>
        <span className={`text-xs px-2 py-1 rounded-full ${
          verificationRate >= 80 ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
          verificationRate >= 60 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
          'bg-destructive/10 text-destructive'
        }`}>
          {verificationRate.toFixed(0)}% Verified
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
        <div className="bg-secondary rounded p-2">
          <div className="text-muted-foreground">Verified</div>
          <div className="font-semibold">{verification.verified_claims}</div>
        </div>
        <div className="bg-secondary rounded p-2">
          <div className="text-muted-foreground">Unverified</div>
          <div className="font-semibold">{verification.unverified_count}</div>
        </div>
        <div className="bg-secondary rounded p-2">
          <div className="text-muted-foreground">Total</div>
          <div className="font-semibold">{verification.total_claims}</div>
        </div>
      </div>

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {verification.claims.map((claim, idx) => (
          <div
            key={idx}
            className={`p-2 rounded text-sm border ${
              claim.verified
                ? 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800'
                : 'bg-destructive/5 border-destructive/30'
            }`}
          >
            <div className="flex items-start gap-2">
              {claim.verified ? (
                <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-4 h-4 text-destructive flex-shrink-0 mt-0.5" />
              )}
              <div className="flex-1">
                <p className="leading-relaxed">{claim.claim}</p>
                <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                  <span>Confidence: {(claim.confidence * 100).toFixed(0)}%</span>
                  {claim.evidence_quality && (
                    <span>Quality: {claim.evidence_quality}</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
