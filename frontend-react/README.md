# EBM RAG React Frontend

Modern React/Next.js frontend for the Evidence-Based Medicine RAG System.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd frontend-react
npm install
```

### 2. Configure Environment

The API URL is already configured in `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

### 3. Start Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## 📋 Prerequisites

**Backend must be running first:**
```bash
# In project root
python backend/run_api.py
```

## 🎨 Features

✅ **Modern UI** - Clean, responsive design with Tailwind CSS  
✅ **Real-time Chat** - Instant messaging interface  
✅ **Gate Visualization** - 3-level energy gate status display  
✅ **Evidence Cards** - Expandable evidence passages with metadata  
✅ **Follow-up Questions** - Clickable suggestion buttons  
✅ **Contradiction Alerts** - Visual warnings for conflicting evidence  
✅ **Evidence Verification** - Sentence-by-sentence validation display  
✅ **Dark Mode** - Automatic theme switching  
✅ **Responsive** - Works on desktop, tablet, and mobile  

## 🏗️ Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **Icons**: Lucide React

## 📁 Project Structure

```
frontend-react/
├── app/
│   ├── globals.css          # Global styles
│   ├── layout.tsx           # Root layout
│   └── page.tsx             # Home page
├── components/
│   ├── ChatInterface.tsx    # Main chat component
│   ├── MessageBubble.tsx    # Message display
│   ├── Sidebar.tsx          # Settings sidebar
│   ├── GateStatus.tsx       # Gate visualization
│   ├── EvidenceCard.tsx     # Evidence display
│   ├── ContradictionAlert.tsx
│   └── EvidenceVerification.tsx
├── lib/
│   └── api.ts               # API client
├── types/
│   └── index.ts             # TypeScript types
└── public/                  # Static assets
```

## 🔧 Configuration

Edit `next.config.js` to change settings:
```javascript
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: 'http://localhost:8000/api/v1',
  },
}
```

## 🚀 Deployment

### Build for Production

```bash
npm run build
npm start
```

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

### Deploy to Netlify

```bash
npm run build
# Upload 'out' folder to Netlify
```

## 🧪 Testing

```bash
# Start both backend and frontend
# Terminal 1
python backend/run_api.py

# Terminal 2
cd frontend-react
npm run dev
```

Then visit http://localhost:3000 and test:
1. Ask "What is hypertension?"
2. View gate status visualization
3. Click follow-up questions
4. Expand evidence cards
5. Check contradiction alerts
6. Review evidence verification

## 🎯 Features Comparison

| Feature | Streamlit | React |
|---------|-----------|-------|
| UI Framework | Streamlit | Next.js + Tailwind |
| Interactivity | Limited | Full control |
| Customization | Moderate | Complete |
| Performance | Good | Excellent |
| Mobile Support | Basic | Full responsive |
| Deployment | Streamlit Cloud | Vercel/Netlify/AWS |

## 📝 License

Part of the EBM RAG System project.
