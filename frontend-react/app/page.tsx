'use client'

import React, { useState } from 'react';
import Link from 'next/link'
import { Sun, Moon, Shield, Database, Brain, AlertCircle } from 'lucide-react';

export default function AskMyCatLanding() {
  const [isDark, setIsDark] = useState(false);

  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDark ? 'bg-gray-900' : 'bg-white'}`}>
      {/* Gradient Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {isDark ? (
          <>
            {/* Synthwave / Cyberpunk inspired gradient */}
            <div className="absolute -top-32 -left-32 w-[800px] h-[800px] bg-purple-600 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob"></div>
            <div className="absolute -top-32 -right-32 w-[800px] h-[800px] bg-pink-500 rounded-full mix-blend-normal filter blur-3xl opacity-45 animate-blob animation-delay-2000"></div>
            
            {/* Middle section - turquoise and blue */}
            <div className="absolute top-1/4 left-1/4 w-[800px] h-[800px] bg-cyan-500 rounded-full mix-blend-normal filter blur-3xl opacity-45 animate-blob animation-delay-4000"></div>
            <div className="absolute top-1/3 right-1/4 w-[800px] h-[800px] bg-blue-600 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob animation-delay-3000"></div>
            <div className="absolute top-1/2 left-1/3 w-[800px] h-[800px] bg-indigo-600 rounded-full mix-blend-normal filter blur-3xl opacity-45 animate-blob animation-delay-5000"></div>
            
            {/* Bottom section - sunset colors */}
            <div className="absolute bottom-1/4 right-1/3 w-[800px] h-[800px] bg-fuchsia-600 rounded-full mix-blend-normal filter blur-3xl opacity-40 animate-blob animation-delay-6000"></div>
            <div className="absolute -bottom-32 left-1/4 w-[800px] h-[800px] bg-violet-600 rounded-full mix-blend-normal filter blur-3xl opacity-45 animate-blob animation-delay-7000"></div>
          </>
        ) : (
          <>
            {/* Distinct colors with less blending - spread apart */}
            <div className="absolute -top-48 -left-48 w-[900px] h-[900px] bg-blue-500 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob"></div>
            <div className="absolute -top-48 -right-48 w-[900px] h-[900px] bg-amber-400 rounded-full mix-blend-normal filter blur-3xl opacity-55 animate-blob animation-delay-2000"></div>
            
            {/* Middle section - more spread */}
            <div className="absolute top-1/4 -left-24 w-[900px] h-[900px] bg-rose-500 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob animation-delay-4000"></div>
            <div className="absolute top-1/3 -right-24 w-[900px] h-[900px] bg-emerald-500 rounded-full mix-blend-normal filter blur-3xl opacity-55 animate-blob animation-delay-3000"></div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 w-[900px] h-[900px] bg-violet-500 rounded-full mix-blend-normal filter blur-3xl opacity-45 animate-blob animation-delay-5000"></div>
            
            {/* Bottom section - spread corners */}
            <div className="absolute -bottom-48 -right-48 w-[900px] h-[900px] bg-orange-500 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob animation-delay-6000"></div>
            <div className="absolute -bottom-48 -left-48 w-[900px] h-[900px] bg-cyan-500 rounded-full mix-blend-normal filter blur-3xl opacity-55 animate-blob animation-delay-7000"></div>
          </>
        )}
      </div>

      {/* Content */}
      <div className="relative z-10">
        {/* Navigation */}
        <nav className="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
          <div className="flex items-center gap-8">
            <a href="#features" className={`font-medium ${isDark ? 'text-white' : 'text-black'} hover:opacity-70 transition-opacity`}>Features</a>
            <a href="#docs" className={`font-medium ${isDark ? 'text-white' : 'text-black'} hover:opacity-70 transition-opacity`}>Docs</a>
            <a href="#about" className={`font-medium ${isDark ? 'text-white' : 'text-black'} hover:opacity-70 transition-opacity`}>About</a>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsDark(!isDark)}
              className={`p-2 rounded-lg ${isDark ? 'bg-gray-800 text-yellow-400' : 'bg-gray-100 text-gray-700'} hover:opacity-80 transition-opacity`}
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <Link href="/signup">
              <button className={`px-6 py-2.5 rounded-full font-medium ${isDark ? 'bg-white text-black' : 'bg-black text-white'} hover:opacity-90 transition-opacity`}>
                Sign up
              </button>
            </Link>
          </div>
        </nav>

        {/* Hero Section */}
        <div className="max-w-6xl mx-auto px-8 pt-16 pb-32 text-center">
          <h1 className={`text-7xl font-bold mb-12 ${isDark ? 'text-white' : 'text-black'}`} style={{ fontFamily: 'monospace', letterSpacing: '-0.02em' }}>
            Medical answers<br />you can trust
          </h1>
          
          <div className={`inline-block px-16 py-16 rounded-3xl backdrop-blur-md ${isDark ? 'bg-white/10 border border-white/20' : 'bg-white/50 border border-white/40'} shadow-2xl max-w-3xl`}>
            <p className={`text-xl mb-10 leading-relaxed ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>
              From evidence retrieval to hallucination detection -<br />
              only speaks when it knows, otherwise it stays silent.
            </p>
            
            <div className="flex items-center justify-center gap-4">
              <Link href="/ask">
                <button className={`px-8 py-4 rounded-full font-medium text-lg ${isDark ? 'bg-white text-black' : 'bg-black text-white'} hover:opacity-90 transition-all hover:scale-105`}>
                  Ask a question
                </button>
              </Link>
              <button className={`px-8 py-4 rounded-full font-medium text-lg ${isDark ? 'bg-white/10 text-white border-2 border-white/30' : 'bg-white border-2 border-gray-300 text-black'} hover:opacity-90 transition-all hover:scale-105`}>
                View demo
              </button>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="max-w-6xl mx-auto px-8 pb-20">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Energy Gating System */}
            <div className={`p-8 rounded-2xl ${isDark ? 'bg-white/5 hover:bg-white/95 border-white/10' : 'bg-white/30 hover:bg-white/95 border-gray-200'} backdrop-blur-sm border hover:shadow-2xl transition-all duration-300 cursor-pointer group hover:scale-105`}>
              <div className={`w-14 h-14 rounded-xl ${isDark ? 'bg-gradient-to-br from-purple-500 to-pink-500' : 'bg-gradient-to-br from-purple-400 to-pink-400'} flex items-center justify-center mb-6 shadow-lg`}>
                <Shield className="w-7 h-7 text-white" />
              </div>
              <h3 className={`text-xl font-bold mb-3 flex items-center gap-2 ${isDark ? 'text-white group-hover:text-gray-900' : 'text-black'} transition-colors`}>
                3-Gate Energy System
                <span className="text-sm">→</span>
              </h3>
              <p className={`${isDark ? 'text-gray-400 group-hover:text-gray-700' : 'text-gray-600'} leading-relaxed transition-colors`}>
                Query quality, retrieval quality, and evidence consistency checks ensure only reliable answers pass through
              </p>
            </div>

            {/* Evidence Verification */}
            <div className={`p-8 rounded-2xl ${isDark ? 'bg-white/5 hover:bg-white/95 border-white/10' : 'bg-white/30 hover:bg-white/95 border-gray-200'} backdrop-blur-sm border hover:shadow-2xl transition-all duration-300 cursor-pointer group hover:scale-105`}>
              <div className={`w-14 h-14 rounded-xl ${isDark ? 'bg-gradient-to-br from-blue-500 to-cyan-500' : 'bg-gradient-to-br from-blue-400 to-cyan-400'} flex items-center justify-center mb-6 shadow-lg`}>
                <Database className="w-7 h-7 text-white" />
              </div>
              <h3 className={`text-xl font-bold mb-3 flex items-center gap-2 ${isDark ? 'text-white group-hover:text-gray-900' : 'text-black'} transition-colors`}>
                Evidence Verification
                <span className="text-sm">→</span>
              </h3>
              <p className={`${isDark ? 'text-gray-400 group-hover:text-gray-700' : 'text-gray-600'} leading-relaxed transition-colors`}>
                Sentence-level verification with citation mapping and hallucination detection for complete transparency
              </p>
            </div>

            {/* Smart Retrieval */}
            <div className={`p-8 rounded-2xl ${isDark ? 'bg-white/5 hover:bg-white/95 border-white/10' : 'bg-white/30 hover:bg-white/95 border-gray-200'} backdrop-blur-sm border hover:shadow-2xl transition-all duration-300 cursor-pointer group hover:scale-105`}>
              <div className={`w-14 h-14 rounded-xl ${isDark ? 'bg-gradient-to-br from-green-500 to-emerald-500' : 'bg-gradient-to-br from-green-400 to-emerald-400'} flex items-center justify-center mb-6 shadow-lg`}>
                <Brain className="w-7 h-7 text-white" />
              </div>
              <h3 className={`text-xl font-bold mb-3 flex items-center gap-2 ${isDark ? 'text-white group-hover:text-gray-900' : 'text-black'} transition-colors`}>
                Neural Reranking
                <span className="text-sm">→</span>
              </h3>
              <p className={`${isDark ? 'text-gray-400 group-hover:text-gray-700' : 'text-gray-600'} leading-relaxed transition-colors`}>
                Custom-trained ranking model with ChromaDB integration for precise medical literature retrieval
              </p>
            </div>

            {/* Contradiction Detection */}
            <div className={`p-8 rounded-2xl ${isDark ? 'bg-white/5 hover:bg-white/95 border-white/10' : 'bg-white/30 hover:bg-white/95 border-gray-200'} backdrop-blur-sm border hover:shadow-2xl transition-all duration-300 cursor-pointer group hover:scale-105`}>
              <div className={`w-14 h-14 rounded-xl ${isDark ? 'bg-gradient-to-br from-orange-500 to-red-500' : 'bg-gradient-to-br from-orange-400 to-red-400'} flex items-center justify-center mb-6 shadow-lg`}>
                <AlertCircle className="w-7 h-7 text-white" />
              </div>
              <h3 className={`text-xl font-bold mb-3 flex items-center gap-2 ${isDark ? 'text-white group-hover:text-gray-900' : 'text-black'} transition-colors`}>
                Conflict Detection
                <span className="text-sm">→</span>
              </h3>
              <p className={`${isDark ? 'text-gray-400 group-hover:text-gray-700' : 'text-gray-600'} leading-relaxed transition-colors`}>
                Identifies and explains contradictions in medical evidence with severity classification
              </p>
            </div>
          </div>
        </div>
      </div>

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
        .animation-delay-6000 {
          animation-delay: 6s;
        }
        .animation-delay-3000 {
          animation-delay: 3s;
        }
        .animation-delay-3500 {
          animation-delay: 3.5s;
        }
        .animation-delay-5000 {
          animation-delay: 5s;
        }
        .animation-delay-7000 {
          animation-delay: 7s;
        }
      `}</style>
    </div>
  );
}
