'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { ChatMessage, QuerySettings } from '@/types'
import { queryRAG } from '@/lib/api'
import MessageBubble from './MessageBubble'

interface ChatInterfaceProps {
  messages: ChatMessage[]
  setMessages: (messages: ChatMessage[]) => void
  settings: QuerySettings
}

export default function ChatInterface({ messages, setMessages, settings }: ChatInterfaceProps) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Listen for example queries
  useEffect(() => {
    const handleExampleQuery = (e: any) => {
      setInput(e.detail)
    }
    window.addEventListener('example-query', handleExampleQuery)
    return () => window.removeEventListener('example-query', handleExampleQuery)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    }

    setMessages([...messages, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await queryRAG(input.trim(), settings)
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        response,
        timestamp: Date.now(),
      }

      setMessages([...messages, userMessage, assistantMessage])
    } catch (error: any) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Error: ${error.response?.data?.detail || error.message || 'Failed to get response'}`,
        timestamp: Date.now(),
      }
      setMessages([...messages, userMessage, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFollowUp = (question: string) => {
    setInput(question)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 space-y-6 mb-6">
        {messages.length === 0 && (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg mb-2">👋 Welcome to EBM RAG System</p>
            <p className="text-sm">Ask a medical question to get started</p>
          </div>
        )}
        
        {messages.map((message, idx) => (
          <MessageBubble 
            key={idx} 
            message={message} 
            onFollowUp={handleFollowUp}
          />
        ))}
        
        {isLoading && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Processing your query...</span>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="relative">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a medical question..."
          disabled={isLoading}
          className="w-full px-4 py-3 pr-12 rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-md bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  )
}
