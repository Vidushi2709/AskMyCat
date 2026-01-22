'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'

export default function SignIn() {
  const router = useRouter()
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // For now, just redirect to chat (you can add real auth later)
    router.push('/chat')
  }

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 flex items-center justify-center px-6">
      {/* Fixed gradient background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-48 -left-48 w-[900px] h-[900px] bg-blue-500 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob"></div>
        <div className="absolute -top-48 -right-48 w-[900px] h-[900px] bg-amber-400 rounded-full mix-blend-normal filter blur-3xl opacity-55 animate-blob animation-delay-2000"></div>
        <div className="absolute top-1/4 -left-24 w-[900px] h-[900px] bg-rose-500 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob animation-delay-4000"></div>
        <div className="absolute top-1/3 -right-24 w-[900px] h-[900px] bg-emerald-500 rounded-full mix-blend-normal filter blur-3xl opacity-55 animate-blob animation-delay-3000"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 w-[900px] h-[900px] bg-violet-500 rounded-full mix-blend-normal filter blur-3xl opacity-45 animate-blob animation-delay-5000"></div>
        <div className="absolute -bottom-48 -right-48 w-[900px] h-[900px] bg-orange-500 rounded-full mix-blend-normal filter blur-3xl opacity-50 animate-blob animation-delay-6000"></div>
        <div className="absolute -bottom-48 -left-48 w-[900px] h-[900px] bg-cyan-500 rounded-full mix-blend-normal filter blur-3xl opacity-55 animate-blob animation-delay-7000"></div>
      </div>
      <div className="w-full max-w-xl relative z-10">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-3" style={{ fontFamily: 'monospace' }}>
            Welcome Back
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">Sign in to access your account</p>
        </div>

        {/* Sign In Form */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl p-6 border border-gray-200 dark:border-gray-700">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Email Address
              </label>
              <input
                id="email"
                type="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
                placeholder="••••••••"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Remember me</span>
              </label>
              <a href="#" className="text-sm text-purple-600 dark:text-purple-400 hover:underline">
                Forgot password?
              </a>
            </div>

            <button
              type="submit"
              className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 transition shadow-lg font-semibold text-lg"
            >
              Sign In
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-gray-600 dark:text-gray-400">
              Don't have an account?{' '}
              <Link href="/signup" className="text-purple-600 dark:text-purple-400 font-semibold hover:underline">
                Sign Up
              </Link>
            </p>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes blob {
          0% {
            transform: translate(0px, 0px) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
          100% {
            transform: translate(0px, 0px) scale(1);
          }
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
