'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, Send } from 'lucide-react'

export default function AskQuestion() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  })
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Here you would typically send the data to your backend
    console.log('Form submitted:', formData)
    setSubmitted(true)
    setTimeout(() => {
      setSubmitted(false)
      setFormData({ name: '', email: '', subject: '', message: '' })
    }, 3000)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:bg-gray-900">
      {/* Fixed gradient background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-48 -left-48 w-[900px] h-[900px] bg-blue-500 rounded-full mix-blend-normal filter blur-3xl opacity-30 animate-blob"></div>
        <div className="absolute -top-48 -right-48 w-[900px] h-[900px] bg-purple-500 rounded-full mix-blend-normal filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-48 left-1/2 w-[900px] h-[900px] bg-pink-500 rounded-full mix-blend-normal filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <div className="max-w-4xl mx-auto px-6 py-8">
          <Link 
            href="/" 
            className="inline-flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition mb-8"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Home
          </Link>

          <div className="text-center mb-12">
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 dark:text-white mb-4" style={{ fontFamily: 'monospace' }}>
              Ask a Question
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400">
              Have a medical question? Send us a message and we'll get back to you.
            </p>
          </div>

          {/* Form Card */}
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl p-8 md:p-12 border border-gray-200 dark:border-gray-700">
            {submitted ? (
              <div className="text-center py-12">
                <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-3">Message Sent!</h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Thank you for your question. We'll get back to you soon.
                </p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Name Field */}
                <div>
                  <label htmlFor="name" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Your Name *
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                    placeholder="John Doe"
                  />
                </div>

                {/* Email Field */}
                <div>
                  <label htmlFor="email" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Email Address *
                  </label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    required
                    value={formData.email}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                    placeholder="john@example.com"
                  />
                </div>

                {/* Subject Field */}
                <div>
                  <label htmlFor="subject" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Subject *
                  </label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    required
                    value={formData.subject}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                    placeholder="What's your question about?"
                  />
                </div>

                {/* Message Field */}
                <div>
                  <label htmlFor="message" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Your Question *
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    required
                    rows={6}
                    value={formData.message}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent transition resize-none"
                    placeholder="Describe your medical question in detail..."
                  />
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  className="w-full px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 transition shadow-lg text-lg font-semibold flex items-center justify-center gap-2"
                >
                  <Send className="w-5 h-5" />
                  Send Message
                </button>

                <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                  We typically respond within 24-48 hours
                </p>
              </form>
            )}
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
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </div>
  )
}
