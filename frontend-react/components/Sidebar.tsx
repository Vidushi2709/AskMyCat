import { Settings, Database, Shield, CheckCircle, AlertTriangle } from 'lucide-react'
import { QuerySettings } from '@/types'

interface SidebarProps {
  settings: QuerySettings
  setSettings: (settings: QuerySettings) => void
}

export default function Sidebar({ settings, setSettings }: SidebarProps) {
  const updateSetting = (key: keyof QuerySettings, value: any) => {
    setSettings({ ...settings, [key]: value })
  }

  const exampleQueries = [
    "What is hypertension?",
    "Treatment for type 2 diabetes",
    "Symptoms of myocardial infarction",
    "What are the side effects of aspirin?",
    "When to use antibiotics?"
  ]

  return (
    <aside className="w-80 border-r bg-card overflow-y-auto">
      <div className="p-6 space-y-6">
        {/* Settings Header */}
        <div className="flex items-center gap-2 text-lg font-semibold">
          <Settings className="w-5 h-5" />
          Configuration
        </div>

        {/* Retrieval Settings */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <Database className="w-4 h-4" />
            Retrieval Settings
          </div>
          
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium flex justify-between mb-2">
                <span>Top-k passages</span>
                <span className="text-muted-foreground">{settings.top_k}</span>
              </label>
              <input
                type="range"
                min="5"
                max="50"
                step="5"
                value={settings.top_k}
                onChange={(e) => updateSetting('top_k', parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-sm font-medium flex justify-between mb-2">
                <span>Similarity threshold</span>
                <span className="text-muted-foreground">{settings.threshold.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.threshold}
                onChange={(e) => updateSetting('threshold', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Advanced Features */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <Shield className="w-4 h-4" />
            Advanced Features
          </div>

          <div className="space-y-3">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.use_llm}
                onChange={(e) => updateSetting('use_llm', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm">Generate LLM Answer</span>
            </label>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.enable_gates}
                onChange={(e) => updateSetting('enable_gates', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm">Enable Energy Gates</span>
            </label>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.verify_evidence}
                onChange={(e) => updateSetting('verify_evidence', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm">Verify Evidence Chain</span>
            </label>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.detect_conflicts}
                onChange={(e) => updateSetting('detect_conflicts', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm">Detect Contradictions</span>
            </label>
          </div>
        </div>

        {/* Example Queries */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-muted-foreground">
            Example Queries
          </div>
          <div className="space-y-2">
            {exampleQueries.map((query, idx) => (
              <button
                key={idx}
                className="w-full text-left text-sm px-3 py-2 rounded-md bg-secondary hover:bg-secondary/80 transition-colors"
                onClick={() => {
                  const event = new CustomEvent('example-query', { detail: query })
                  window.dispatchEvent(event)
                }}
              >
                {query}
              </button>
            ))}
          </div>
        </div>
      </div>
    </aside>
  )
}
