'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import ChatInterface from '@/components/ChatInterface'
import { ChatMessage, QuerySettings } from '@/types'
import { LogOut, Edit, Menu, X, Sun, Moon } from 'lucide-react'

export default function ChatPage() {
  const router = useRouter()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isDark, setIsDark] = useState(false)
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

  const handleNewChat = () => {
    setMessages([])
  }

  return (
    <div className={`flex h-screen ${isDark ? 'bg-gray-900' : 'bg-white'} relative`}>
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
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 border-r ${isDark ? 'border-gray-700 bg-gray-900/50' : 'border-gray-200 bg-white/50'} flex flex-col backdrop-blur-sm relative z-10`}>
        {sidebarOpen && (
          <>
            {/* New Chat Button */}
            <div className="p-3">
              <div className="flex items-center justify-between gap-2">
                <button
                  onClick={handleNewChat}
                  className={`flex-1 flex items-center gap-2 px-4 py-3 ${isDark ? 'text-gray-300 hover:bg-gray-800' : 'text-gray-700 hover:bg-gray-100'} rounded-lg transition font-medium`}
                >
                  <Edit size={18} />
                  New chat
                </button>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className={`p-2 ${isDark ? 'text-gray-400 hover:text-gray-100' : 'text-gray-600 hover:text-gray-900'} transition`}
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            {/* Empty space for future features */}
            <div className="flex-1"></div>

            {/* User Section */}
            <div className={`p-3 border-t ${isDark ? 'border-gray-700' : 'border-gray-200'}`}>
              <button
                onClick={handleLogout}
                className={`w-full flex items-center gap-2 px-4 py-2 ${isDark ? 'text-gray-300 hover:bg-gray-800' : 'text-gray-700 hover:bg-gray-100'} rounded-lg transition text-sm`}
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
        <header className={`border-b ${isDark ? 'border-gray-700 bg-gray-900/50' : 'border-gray-200 bg-white/50'} backdrop-blur-sm px-6 py-4`}>
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              {!sidebarOpen && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  className={`${isDark ? 'text-gray-400 hover:text-gray-100' : 'text-gray-600 hover:text-gray-900'} transition`}
                >
                  <Menu size={24} />
                </button>
              )}
              <div>
                <h1 className={`text-xl font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Evidence-Based Medicine RAG
                </h1>
                <p className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-500'} mt-0.5`}>
                  Multi-Level Gating • Hallucination Prevention • Evidence Verification
                </p>
              </div>
            </div>
            
            {/* Theme Toggle */}
            <button
              onClick={() => setIsDark(!isDark)}
              className={`p-2 rounded-lg ${isDark ? 'bg-gray-800 text-yellow-400' : 'bg-gray-100 text-gray-700'} hover:opacity-80 transition-opacity`}
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div className={`flex-1 overflow-y-auto ${isDark ? 'bg-gray-900/30' : 'bg-white/30'} backdrop-blur-sm`}>
          <div className="max-w-4xl mx-auto px-6 py-6 w-full">

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
