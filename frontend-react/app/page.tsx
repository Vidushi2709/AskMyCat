'use client'

import Link from 'next/link'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-white to-pink-100 dark:from-gray-900 dark:via-gray-900 dark:to-purple-900/30">
      {/* Navbar */}
      <nav className="border-b bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🏥</span>
              <span className="text-xl font-bold bg-gradient-to-r from-pink-500 to-purple-600 bg-clip-text text-transparent">
                EBM RAG
              </span>
            </div>
            <div className="flex items-center gap-4">
              <Link 
                href="/signin" 
                className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition"
              >
                Sign In
              </Link>
              <Link 
                href="/signup" 
                className="px-6 py-2 bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-lg hover:from-pink-600 hover:to-purple-700 transition shadow-md"
              >
                Sign Up
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center space-y-8">
          <div className="space-y-4">
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 dark:text-white">
              Evidence-Based Medicine
              <br />
              <span className="bg-gradient-to-r from-pink-500 to-purple-600 bg-clip-text text-transparent">
                RAG System
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              Advanced medical research assistant powered by AI. Get accurate, evidence-based answers with source verification.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Link 
              href="/signup" 
              className="px-8 py-4 bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-xl hover:from-pink-600 hover:to-purple-700 transition shadow-lg text-lg font-semibold"
            >
              Get Started
            </Link>
            <Link 
              href="/signin" 
              className="px-8 py-4 border-2 border-purple-500 text-purple-600 dark:text-purple-400 rounded-xl hover:bg-purple-50 dark:hover:bg-purple-900/20 transition text-lg font-semibold"
            >
              Sign In
            </Link>
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-8 pt-16">
            <div className="p-6 bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">Multi-Level Gating</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Advanced filtering system ensures only relevant and accurate information reaches you
              </p>
            </div>
            <div className="p-6 bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-4xl mb-4">✅</div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">Evidence Verification</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Every answer is backed by verified medical literature and research papers
              </p>
            </div>
            <div className="p-6 bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-4xl mb-4">🛡️</div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">Hallucination Prevention</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Built-in safeguards prevent AI from generating unverified medical information
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer with gradient */}
      <footer className="mt-20 py-8 text-center text-gray-600 dark:text-gray-400">
        <p>© 2026 EBM RAG System. All rights reserved.</p>
      </footer>
    </div>
  )
}
