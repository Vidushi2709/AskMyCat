'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { 
  Sun, Moon, ArrowLeft, CheckCircle, Shield, Database, Brain, 
  AlertCircle, Zap, Lock, Eye, BookOpen, Filter, Layers, TrendingUp,
  Sparkles, Target
} from 'lucide-react'

export default function FeaturesPage() {
  const [isDark, setIsDark] = useState(false)
  const [expandedFeature, setExpandedFeature] = useState<number | null>(null)

  const features = [
    {
      icon: Brain,
      title: "Evidence-Based Medicine Integration",
      description: "Leverages curated medical evidence and clinical guidelines",
      details: "Integrates with large medical knowledge bases to provide answers grounded in peer-reviewed research and clinical evidence. Every response is backed by verified sources.",
      color: "from-blue-400 to-blue-600"
    },
    {
      icon: Database,
      title: "Vector-Based Retrieval",
      description: "Fast semantic search through medical documents",
      details: "Uses advanced embeddings and ChromaDB to perform semantic search across medical literature, finding relevant passages based on meaning rather than keywords.",
      color: "from-purple-400 to-purple-600"
    },
    {
      icon: Shield,
      title: "Multi-Level Gating System",
      description: "Prevents hallucinations with intelligent filtering",
      details: "Implements a 3-layer gating mechanism (EBM Filter, Energy-Based Score, Contradiction Detector) to ensure responses are factually grounded and don't exceed the knowledge base.",
      color: "from-emerald-400 to-emerald-600"
    },
    {
      icon: Zap,
      title: "Energy-Based Quality Scoring",
      description: "Quantifies passage quality and relevance",
      details: "Uses energy-based models to score passages on quality, specificity, and relevance. Lower energy = higher quality responses. Ensures only the best evidence is used.",
      color: "from-yellow-400 to-yellow-600"
    },
    {
      icon: Eye,
      title: "Source Attribution & Citations",
      description: "Transparent source tracking and citations",
      details: "All responses include numbered citations with metadata (source title, publication year, PMID/URL). Users can verify claims by reviewing original sources.",
      color: "from-pink-400 to-pink-600"
    },
    {
      icon: AlertCircle,
      title: "Hallucination Prevention",
      description: "5-layer defense against AI fabrications",
      details: "Detects when the LLM tries to claim facts not in the evidence, with early exits for identical passages, temperature control (0.0), evidence validation, and post-response verification.",
      color: "from-red-400 to-red-600"
    },
    {
      icon: Lock,
      title: "Deterministic & Safe Responses",
      description: "Reproducible answers with controlled randomness",
      details: "Temperature set to 0.0 for consistent, deterministic outputs. No creative embellishments or random variations that could lead to false claims.",
      color: "from-indigo-400 to-indigo-600"
    },
    {
      icon: Sparkles,
      title: "Warm & Professional Tone",
      description: "Sounds like a trusted healthcare provider",
      details: "Carefully crafted system prompts make the AI respond as a warm, efficient, and professional healthcare advisor. Clear communication without jargon overload.",
      color: "from-cyan-400 to-cyan-600"
    },
    {
      icon: BookOpen,
      title: "Follow-Up Question Generation",
      description: "Guides users to explore related topics",
      details: "Automatically generates relevant follow-up questions based on the answer, helping users explore the topic deeper and understand context.",
      color: "from-orange-400 to-orange-600"
    },
    {
      icon: Filter,
      title: "Intelligent Evidence Filtering",
      description: "Ranks and filters passages for relevance",
      details: "Uses BM25 ranking and similarity thresholding to filter irrelevant passages. Only top-k most relevant passages are passed to the LLM.",
      color: "from-teal-400 to-teal-600"
    },
    {
      icon: Layers,
      title: "Configurable Query Parameters",
      description: "Fine-tune retrieval and filtering behavior",
      details: "Users can adjust top-k passages, similarity thresholds, enable/disable gates, and configure evidence verification to suit their needs.",
      color: "from-fuchsia-400 to-fuchsia-600"
    },
    {
      icon: TrendingUp,
      title: "Performance Optimization",
      description: "Fast responses with caching and indexing",
      details: "Uses disk-based caching and optimized vector indices to deliver responses in seconds. Scales efficiently to large medical knowledge bases.",
      color: "from-green-400 to-green-600"
    }
  ]

  return (
    <div className="min-h-screen transition-colors duration-300 bg-background">
      {/* Gradient Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {isDark ? (
          <>
            <div className="absolute -top-32 -left-32 w-[800px] h-[800px] bg-purple-600 rounded-full mix-blend-normal filter blur-3xl opacity-30 animate-blob"></div>
            <div className="absolute -top-32 -right-32 w-[800px] h-[800px] bg-pink-500 rounded-full mix-blend-normal filter blur-3xl opacity-25 animate-blob animation-delay-2000"></div>
            <div className="absolute top-1/4 left-1/4 w-[800px] h-[800px] bg-cyan-500 rounded-full mix-blend-normal filter blur-3xl opacity-25 animate-blob animation-delay-4000"></div>
          </>
        ) : (
          <>
            <div className="absolute -top-48 -left-48 w-[900px] h-[900px] bg-blue-200 rounded-full mix-blend-normal filter blur-3xl opacity-30 animate-blob"></div>
            <div className="absolute -top-48 -right-48 w-[900px] h-[900px] bg-purple-200 rounded-full mix-blend-normal filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
            <div className="absolute top-1/4 left-1/3 w-[800px] h-[800px] bg-pink-200 rounded-full mix-blend-normal filter blur-3xl opacity-25 animate-blob animation-delay-4000"></div>
          </>
        )}
      </div>

      <div className="relative z-10">
        {/* Header Navigation */}
        <nav className="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
          <Link href="/" className="flex items-center gap-2 text-xl font-bold hover:opacity-80 transition-opacity">
            <ArrowLeft className="w-5 h-5" />
            Back
          </Link>
          <button
            onClick={() => setIsDark(!isDark)}
            className={`p-2 rounded-lg transition-colors bg-primary/10 hover:bg-primary/20`}
          >
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </nav>

        {/* Hero Section */}
        <div className="max-w-7xl mx-auto px-8 py-16">
          <div className="text-center mb-16">
            <h1 className="text-5xl font-bold mb-4 text-foreground">
              RAG System Features
            </h1>
            <p className="text-xl text-muted-foreground">
              Comprehensive capabilities for Evidence-Based Medicine retrieval and AI-powered healthcare insights
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon
              const isExpanded = expandedFeature === index

              return (
                <div
                  key={index}
                  onClick={() => setExpandedFeature(isExpanded ? null : index)}
                  className={`cursor-pointer rounded-xl p-6 transition-all duration-300 transform hover:scale-105 ${
                    isDark
                      ? `bg-gray-800 hover:bg-gray-700 border border-gray-700 ${
                          isExpanded ? 'ring-2 ring-blue-500' : ''
                        }`
                      : `bg-white hover:shadow-xl shadow-lg border border-gray-100 ${
                          isExpanded ? 'ring-2 ring-blue-500' : ''
                        }`
                  }`}
                >
                  {/* Icon with gradient background */}
                  <div className={`w-14 h-14 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4`}>
                    <Icon className="w-7 h-7 text-white" />
                  </div>

                  {/* Title and Description */}
                  <h3 className={`text-lg font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {feature.title}
                  </h3>
                  <p className={`text-sm mb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    {feature.description}
                  </p>

                  {/* Expanded Details */}
                  {isExpanded && (
                    <div
                      className={`mt-4 pt-4 border-t ${
                        isDark ? 'border-gray-700' : 'border-gray-200'
                      } animate-in fade-in duration-300`}
                    >
                      <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                        {feature.details}
                      </p>
                      <div className="mt-4 flex items-center gap-2 text-blue-500">
                        <CheckCircle className="w-4 h-4" />
                        <span className="text-sm font-medium">Click to collapse</span>
                      </div>
                    </div>
                  )}

                  {/* Expand Indicator */}
                  {!isExpanded && (
                    <div className={`text-xs font-medium ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                      Click to learn more
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* CTA Section */}
          <div className="mt-20 text-center">
            <div className={`rounded-2xl p-12 ${isDark ? 'bg-gray-800' : 'bg-gradient-to-r from-blue-50 to-purple-50'} border ${isDark ? 'border-gray-700' : 'border-gray-200'}`}>
              <h2 className={`text-3xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Ready to Experience EBM RAG?
              </h2>
              <p className={`text-lg mb-8 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                Start asking evidence-based medical questions with confidence
              </p>
              <div className="flex gap-4 justify-center flex-wrap">
                <Link
                  href="/chat"
                  className="px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Go to Chat
                </Link>
                <Link
                  href="/ask"
                  className={`px-8 py-3 rounded-lg font-semibold transition-colors ${
                    isDark
                      ? 'bg-gray-700 text-white hover:bg-gray-600'
                      : 'bg-white text-blue-600 border-2 border-blue-600 hover:bg-blue-50'
                  }`}
                >
                  Ask a Question
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className={`mt-20 py-8 border-t ${isDark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}>
          <div className="max-w-7xl mx-auto px-8 text-center">
            <p className={`${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Evidence-Based Medicine Retrieval-Augmented Generation System
            </p>
          </div>
        </footer>
      </div>

      {/* CSS for animations */}
      <style jsx>{`
        @keyframes blob {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-3000 {
          animation-delay: 3s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        .animation-delay-5000 {
          animation-delay: 5s;
        }
        .animation-delay-6000 {
          animation-delay: 6s;
        }
        .animation-delay-7000 {
          animation-delay: 7s;
        }
      `}</style>
    </div>
  )
}
