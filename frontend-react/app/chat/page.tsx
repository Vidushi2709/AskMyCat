'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import ChatInterface from '@/components/ChatInterface'
import { ChatMessage, QuerySettings } from '@/types'
import { LogOut, Edit, Menu, X } from 'lucide-react'

export default function ChatPage() {
  const router = useRouter()
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

  const handleNewChat = () => {
    setMessages([])
  }

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 border-r border-gray-200 flex flex-col bg-white`}>
        {sidebarOpen && (
          <>
            {/* New Chat Button */}
            <div className="p-3">
              <button
                onClick={handleNewChat}
                className="w-full flex items-center gap-2 px-4 py-3 text-gray-700 hover:bg-gray-100 rounded-lg transition font-medium"
              >
                <Edit size={18} />
                New chat
              </button>
            </div>

            {/* Empty space for future features */}
            <div className="flex-1"></div>

            {/* User Section */}
            <div className="p-3 border-t border-gray-200">
              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition text-sm"
              >
                <LogOut size={18} />
                Logout
              </button>
            </div>
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white px-6 py-4">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="text-gray-600 hover:text-gray-900 transition"
              >
                {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
              </button>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  Evidence-Based Medicine RAG
                </h1>
                <p className="text-xs text-gray-500 mt-0.5">
                  Multi-Level Gating • Hallucination Prevention • Evidence Verification
                </p>
              </div>
            </div>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto bg-white flex items-center">
          <div className="max-w-4xl mx-auto px-6 py-6 w-full">
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
