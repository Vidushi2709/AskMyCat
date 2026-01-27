'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import ChatInterface from '@/components/ChatInterface'
import { ChatMessage, QuerySettings } from '@/types'
import { useDarkMode } from '@/context/DarkModeContext'
import { LogOut, Menu, X, Sparkles } from 'lucide-react'

export default function ChatPage() {
  const router = useRouter()
  const { isDark, setIsDark } = useDarkMode()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [settings] = useState<QuerySettings>({
    top_k: 10,
    threshold: 0.5,
    enable_gates: true,
    verify_evidence: false,
    detect_conflicts: true,
    use_llm: true,
  })

  const handleLogout = () => {
    router.push('/')
  }

  return (
    <div className={`flex h-screen ${isDark ? 'dark' : ''}`}>
      {/* Lighter gradient background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {isDark ? (
          <>
            <div className="absolute -top-48 -left-48 w-[800px] h-[800px] bg-purple-600 rounded-full mix-blend-normal filter blur-3xl opacity-15 animate-blob"></div>
            <div className="absolute -top-48 -right-48 w-[800px] h-[800px] bg-pink-500 rounded-full mix-blend-normal filter blur-3xl opacity-15 animate-blob animation-delay-2000"></div>
            <div className="absolute -bottom-48 left-1/2 w-[800px] h-[800px] bg-blue-600 rounded-full mix-blend-normal filter blur-3xl opacity-15 animate-blob animation-delay-4000"></div>
          </>
        ) : (
          <>
            <div className="absolute -top-48 -left-48 w-[800px] h-[800px] bg-blue-500 rounded-full mix-blend-normal filter blur-3xl opacity-20 animate-blob"></div>
            <div className="absolute -top-48 -right-48 w-[800px] h-[800px] bg-pink-400 rounded-full mix-blend-normal filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
            <div className="absolute -bottom-48 left-1/2 w-[800px] h-[800px] bg-purple-400 rounded-full mix-blend-normal filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
          </>
        )}
      </div>

      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 border-r border-border bg-card/50 flex flex-col backdrop-blur-sm relative z-10`}>
        {sidebarOpen && (
          <>
            {/* Empty space for future features */}
            <div className="flex-1"></div>

            {/* User Section */}
            <div className="p-3 border-t border-border">
              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-2 px-4 py-2 text-foreground hover:bg-muted/50 rounded-lg transition text-sm"
              >
                <LogOut size={18} />
                Logout
              </button>
            </div>
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden relative z-10">
        {/* Header */}
        <header className="border-b border-border bg-card/50 backdrop-blur-sm px-6 py-4">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              {!sidebarOpen && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="text-foreground/70 hover:text-foreground transition"
                >
                  <Menu size={24} />
                </button>
              )}
              <div>
                <h1 className="text-xl font-semibold text-foreground">
                  Evidence-Based Medicine RAG
                </h1>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Multi-Level Gating • Hallucination Prevention • Evidence Verification
                </p>
              </div>
            </div>
            
            <Link href="/features">
              <button className="p-2 rounded-lg flex items-center gap-2 bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
                <Sparkles className="w-5 h-5" />
                <span className="text-sm font-medium hidden sm:inline">Features</span>
              </button>
            </Link>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto bg-background/50 backdrop-blur-sm">
          <div className="max-w-4xl mx-auto px-6 py-2 w-full">

      <style jsx>{`
        @keyframes blob {
          0% {
            transform: translate(0px, 0px) scale(1);
          }
          33% {
            transform: translate(50px, -80px) scale(1.15);
          }
          66% {
            transform: translate(-40px, 40px) scale(0.85);
          }
          100% {
            transform: translate(0px, 0px) scale(1);
          }
        }
        .animate-blob {
          animation: blob 8s infinite ease-in-out;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
            <ChatInterface 
              messages={messages} 
              setMessages={setMessages}
              settings={settings}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
