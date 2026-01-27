import { CheckCircle, XCircle } from 'lucide-react'
import { EvidenceChain } from '@/types'

interface EvidenceVerificationProps {
  evidenceChain: EvidenceChain
}

export default function EvidenceVerification({ evidenceChain }: EvidenceVerificationProps) {
  const verificationRate = evidenceChain.verification_rate * 100

  return (
    <div className="bg-card border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-sm">Evidence Verification</h4>
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
          <div className="font-semibold">{evidenceChain.verified_sentences}</div>
        </div>
        <div className="bg-secondary rounded p-2">
          <div className="text-muted-foreground">Unverified</div>
          <div className="font-semibold">{evidenceChain.unverified_sentences}</div>
        </div>
        <div className="bg-secondary rounded p-2">
          <div className="text-muted-foreground">Total</div>
          <div className="font-semibold">
            {evidenceChain.verified_sentences + evidenceChain.unverified_sentences}
          </div>
        </div>
      </div>

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {evidenceChain.sentences.map((sentence, idx) => (
          <div
            key={idx}
            className={`p-2 rounded text-sm border ${
              sentence.verified
                ? 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800'
                : 'bg-destructive/5 border-destructive/30'
            }`}
          >
            <div className="flex items-start gap-2">
              {sentence.verified ? (
                <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-4 h-4 text-destructive flex-shrink-0 mt-0.5" />
              )}
              <div className="flex-1">
                <p className="leading-relaxed">{sentence.sentence}</p>
                <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                  <span>Confidence: {(sentence.confidence * 100).toFixed(0)}%</span>
                  {sentence.citations.length > 0 && (
                    <span>Citations: [{sentence.citations.join(', ')}]</span>
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
