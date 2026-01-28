'use client'

import { QuerySettings, EvidenceLevel } from '@/types'
import { Settings, HelpCircle } from 'lucide-react'
import { useState } from 'react'

interface SettingsPanelProps {
  settings: QuerySettings
  onSettingsChange: (settings: QuerySettings) => void
}

export default function SettingsPanel({ settings, onSettingsChange }: SettingsPanelProps) {
  const [isOpen, setIsOpen] = useState(false)

  const handleEvidenceLevelChange = (level: EvidenceLevel) => {
    onSettingsChange({ ...settings, min_evidence_level: level })
  }

  const handleTopKChange = (value: number) => {
    onSettingsChange({ ...settings, top_k: Math.max(1, Math.min(50, value)) })
  }

  const handleToggle = (key: 'use_pubmed_fallback' | 'verify_answer', value: boolean) => {
    onSettingsChange({ ...settings, [key]: value })
  }

  const evidenceLevels: { level: EvidenceLevel; description: string; icon: string }[] = [
    {
      level: 'HIGH',
      description: 'RCTs, systematic reviews - Most reliable',
      icon: '🟢',
    },
    {
      level: 'MODERATE',
      description: 'RCTs with limitations, strong observational - Balanced',
      icon: '🔵',
    },
    {
      level: 'LOW',
      description: 'Observational studies, case-control - Permissive',
      icon: '🟡',
    },
    {
      level: 'VERY_LOW',
      description: 'Case reports, expert opinion - Exploratory',
      icon: '🔴',
    },
  ]

  return (
    <div className="space-y-4">
      {/* Settings Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-2 rounded-lg border border-border bg-card hover:bg-muted transition-colors"
      >
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4" />
          <span className="text-sm font-medium">Query Settings</span>
        </div>
        <span className="text-xs text-muted-foreground">
          {isOpen ? '▼' : '▶'}
        </span>
      </button>

      {/* Settings Panel */}
      {isOpen && (
        <div className="space-y-4 p-4 rounded-lg border border-border bg-card">
          {/* Evidence Level */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 mb-2">
              <label className="text-sm font-semibold">Minimum Evidence Level</label>
              <div
                className="group relative inline-block cursor-help"
                title="Higher levels require more rigorous studies"
              >
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-2">
              {evidenceLevels.map(({ level, description, icon }) => (
                <button
                  key={level}
                  onClick={() => handleEvidenceLevelChange(level)}
                  className={`p-3 rounded-lg border-2 text-left transition-all ${
                    settings.min_evidence_level === level
                      ? 'border-primary bg-primary/10'
                      : 'border-border bg-muted/50 hover:bg-muted'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <span className="text-lg">{icon}</span>
                    <div>
                      <p className="font-medium text-sm">{level}</p>
                      <p className="text-xs text-muted-foreground">{description}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Top K */}
          <div className="space-y-2 pt-2 border-t border-border">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold">Evidence Count</label>
              <span className="text-sm font-medium text-primary">{settings.top_k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="50"
              value={settings.top_k}
              onChange={(e) => handleTopKChange(parseInt(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              More evidence = slower but more comprehensive (1-50 passages)
            </p>
          </div>

          {/* Toggles */}
          <div className="space-y-3 pt-2 border-t border-border">
            {/* PubMed Fallback */}
            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-sm font-medium">PubMed Fallback</span>
              <div className="relative">
                <input
                  type="checkbox"
                  checked={settings.use_pubmed_fallback}
                  onChange={(e) => handleToggle('use_pubmed_fallback', e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-10 h-6 bg-muted rounded-full peer-checked:bg-primary transition-colors"></div>
                <div className="absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform peer-checked:translate-x-4"></div>
              </div>
            </label>
            <p className="text-xs text-muted-foreground ml-1">
              Search PubMed if local evidence insufficient
            </p>

            {/* Verification */}
            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-sm font-medium">Verify Answer</span>
              <div className="relative">
                <input
                  type="checkbox"
                  checked={settings.verify_answer}
                  onChange={(e) => handleToggle('verify_answer', e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-10 h-6 bg-muted rounded-full peer-checked:bg-primary transition-colors"></div>
                <div className="absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform peer-checked:translate-x-4"></div>
              </div>
            </label>
            <p className="text-xs text-muted-foreground ml-1">
              Verify claims against evidence using BiomedNLI (adds ~1 second)
            </p>
          </div>

          {/* Summary */}
          <div className="pt-2 border-t border-border bg-muted/50 rounded p-2">
            <p className="text-xs text-muted-foreground">
              <span className="font-medium">Current Configuration:</span>
              {settings.min_evidence_level} evidence, {settings.top_k} passages,
              {settings.verify_answer ? ' with' : ' without'} verification
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
