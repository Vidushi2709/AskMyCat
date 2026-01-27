'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'

export default function SignUp() {
  const router = useRouter()
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // For now, just redirect to chat (you can add real auth later)
    if (formData.password === formData.confirmPassword) {
      router.push('/chat')
    } else {
      alert('Passwords do not match!')
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-6">
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
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-3" style={{ fontFamily: 'monospace' }}>
            Create Account
          </h1>
          <p className="text-lg text-muted-foreground">Join us to access evidence-based medical answers</p>
        </div>

        {/* Sign Up Form */}
        <div className="bg-card/80 backdrop-blur-xl rounded-3xl shadow-2xl p-6 border border-border">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-foreground mb-2">
                Full Name
              </label>
              <input
                id="name"
                type="text"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground focus:ring-2 focus:ring-ring focus:border-transparent outline-none"
                placeholder="John Doe"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-foreground mb-2">
                Email Address
              </label>
              <input
                id="email"
                type="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground focus:ring-2 focus:ring-ring focus:border-transparent outline-none"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-foreground mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground focus:ring-2 focus:ring-ring focus:border-transparent outline-none"
                placeholder="••••••••"
              />
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-foreground mb-2">
                Confirm Password
              </label>
              <input
                id="confirmPassword"
                type="password"
                required
                value={formData.confirmPassword}
                onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground focus:ring-2 focus:ring-ring focus:border-transparent outline-none"
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              className="w-full px-6 py-4 bg-primary text-primary-foreground rounded-xl hover:opacity-90 transition shadow-lg font-semibold text-lg"
            >
              Create Account
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-muted-foreground">
              Already have an account?{' '}
              <Link href="/signin" className="text-primary font-semibold hover:underline">
                Sign In
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
