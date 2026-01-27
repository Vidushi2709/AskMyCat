'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'

interface DarkModeContextType {
  isDark: boolean
  setIsDark: (isDark: boolean) => void
}

const DarkModeContext = createContext<DarkModeContextType | undefined>(undefined)

export function DarkModeProvider({ children }: { children: React.ReactNode }) {
  const [isDark, setIsDark] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    // Load from localStorage on mount
    const savedDarkMode = localStorage.getItem('ebm-dark-mode')
    if (savedDarkMode !== null) {
      setIsDark(JSON.parse(savedDarkMode))
    } else {
      // Check system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      setIsDark(prefersDark)
    }
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return
    
    // Save to localStorage
    localStorage.setItem('ebm-dark-mode', JSON.stringify(isDark))
    
    // Update document class for Tailwind dark mode
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDark, mounted])

  return (
    <DarkModeContext.Provider value={{ isDark, setIsDark }}>
      {children}
    </DarkModeContext.Provider>
  )
}

export function useDarkMode() {
  const context = useContext(DarkModeContext)
  if (!context) {
    throw new Error('useDarkMode must be used within DarkModeProvider')
  }
  return context
}
