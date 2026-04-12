# CV AI — Intelligent Recruiter Platform

A full-stack AI-powered recruitment platform that extracts, indexes, and intelligently matches candidates from PDF CVs. Built with Next.js 16 and FastAPI.

---

## Features

**CV Ingestion**
- Drag & drop PDF upload with duplicate detection
- AI extraction via Google Gemini (name, skills, experience, education, work history)
- SHA-256 hash deduplication — never indexes the same CV twice
- Quality scoring (excellent / good / fair / poor) per candidate
- Flags bad extractions automatically

**Candidate Search**
- Natural language hybrid search (semantic + keyword)
- Filters: skills, experience, education, seniority, boolean query
- AI fit score per candidate
- Fit breakdown bars (technical, experience, skills)
- Missing skills detection

**Candidate Profiles**
- Full profile with work experience timeline
- Education, certifications, skills
- Capability radar chart
- AI insights via Gemini: summary, strengths, weaknesses, interview questions
- Similar candidates panel

**Analytics**
- Gap Analysis: your candidate pool vs. job requisitions
- Pipeline Health: supply vs. demand by seniority
- Skill distribution radar and ranked list
- Industry breakdown pie and bar charts
- Interactive tech clusters — click to search or export to outreach
- Global filter bar: date range, target role, location, remote status
- CSV export

**Outreach & Email**
- Shortlist candidates from search or candidates page
- Bulk email with personalisation ({name}, {skills})
- 10 professional email templates
- CC / BCC fields, email signature, schedule send
- Draft saving, live preview per recipient
- Gmail OAuth2 integration — real email delivery

**UX**
- Cmd+K command palette
- Page transitions (fade/slide)
- Collapsible sidebar
- Mobile responsive (hamburger drawer)
- Tooltips on all icons
- Dark / light mode
- Glassmorphism design system

---

## Tech Stack

| Layer       | Technology                                      |
|-------------|------------------------------------------------|
| Frontend    | Next.js 16, TypeScript, Tailwind CSS           |
| Components  | shadcn/ui, Framer Motion, Recharts             |
| State       | Zustand (shortlist), React hooks               |
| Backend     | FastAPI, Python 3.11                           |
| AI          | Google Gemini (extraction + insights)          |
| Embeddings  | sentence-transformers/all-MiniLM-L6-v2         |
| Vector DB   | ChromaDB                                       |
| Email       | Gmail API (OAuth2)                             |
| Auth        | JWT (HS256)                                    |

---

## Project Structure

```
cv_ai/
├── app/
│   ├── api/
│   │   ├── auth.py              # JWT auth
│   │   ├── deps.py              # Auth dependencies
│   │   ├── main.py              # FastAPI app
│   │   └── routes/
│   │       ├── ingest.py        # CV upload + duplicate detection
│   │       ├── search.py        # Hybrid search
│   │       ├── candidate.py     # Candidate CRUD + insights
│   │       ├── analytics.py     # Analytics endpoints
│   │       └── email.py         # Email send + schedule
│   ├── email/
│   │   ├── sender.py            # Gmail API sender
│   │   └── tracker.py           # SQLite email history
│   ├── ingest/
│   │   ├── extractor.py         # Gemini + rule-based extraction
│   │   ├── indexer.py           # ChromaDB indexing
│   │   └── parser.py            # PDF parsing (pymupdf)
│   ├── insights/
│   │   └── generator.py         # Gemini insights generation
│   ├── storage/
│   │   └── vector_store.py      # ChromaDB singleton
│   ├── utils/
│   │   ├── encryption.py        # Fernet email encryption
│   │   └── audit.py             # Audit logging
│   └── analytics.py             # Intelligence functions
├── config.py
├── main.py                      # CLI entry point
└── .env

cv-ai-frontend/
├── src/
│   ├── app/
│   │   ├── (auth)/login/
│   │   └── (dashboard)/
│   │       ├── layout.tsx       # Sidebar + Topbar + Cmd+K
│   │       ├── page.tsx         # Dashboard
│   │       ├── upload/          # CV Upload
│   │       ├── candidates/      # Candidates list + [id] profile
│   │       ├── search/          # Search page
│   │       ├── outreach/        # Shortlist + email
│   │       ├── analytics/       # Analytics
│   │       └── settings/        # Settings
│   ├── components/
│   │   ├── layout/              # Sidebar, Topbar, CommandPalette
│   │   ├── dashboard/           # MetricCard, Charts, etc.
│   │   ├── search/              # SearchBar, FilterSidebar, CandidateCard
│   │   └── outreach/            # EmailModal, ShortlistCard
│   ├── hooks/                   # useDashboard, useSearch, useCandidate, etc.
│   ├── services/                # API clients
│   ├── stores/                  # Zustand stores
│   └── types/                   # TypeScript types
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google Cloud project with Gemini API enabled
- Gmail API credentials (for email)

### Backend

```bash
cd cv_ai
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Create `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
PREFERRED_MODEL=gemini-3.1-flash-lite-preview
SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_fernet_key
EMAIL_TRACKER_DB=email_tracker.db
```

Authenticate Gmail:
```bash
python -c "
from app.email.sender import get_sender
from config import config
sender = get_sender(config.CREDENTIALS_PATH, config.TOKEN_PATH)
sender.authenticate()
print('Gmail authenticated!')
"
```

Start backend:
```bash
python -m uvicorn app.api.main:app --reload --port 8000
```

### Frontend

```bash
cd cv-ai-frontend
npm install
npm run dev
```

Open `http://localhost:3000`

**Default credentials:** `admin` / `secret`

---

## Usage

### Ingest CVs

**Via UI:** Go to `/upload` → drag and drop PDFs → click Upload

**Via CLI:**
```bash
python main.py ingest path/to/cvs/
```

### Search

Natural language queries work best:
- `"senior React developer with 5 years experience"`
- `"Python data scientist machine learning"`
- `"DevOps engineer Kubernetes AWS"`

### Analytics

Go to `/analytics` → select a job requisition → Gap Analysis shows where your pool is strong and where you need to source more talent.

### Outreach

1. Search for candidates
2. Click **Shortlist** on candidate cards
3. Go to `/outreach`
4. Select candidates → **Email** → choose template → **Send**

---

## API Endpoints

```
POST /token                         Login (form-data)
GET  /health                        Health check

POST /api/ingest                    Upload + index CV
GET  /api/ingest/stats              DB stats
DELETE /api/ingest/{id}             Delete candidate

GET  /api/search                    Hybrid search
GET  /api/candidates                List all candidates
GET  /api/candidate/{id}            Get candidate profile
POST /api/candidate/{id}/insights   Generate AI insights

POST /api/email                     Send or schedule email

GET  /api/analytics/overview        Dashboard metrics
GET  /api/analytics/clusters        Tech clusters
GET  /api/analytics/similar/{id}    Similar candidates
GET  /api/analytics/enrich/{id}     Enriched profile
```

---

## Environment Variables

| Variable            | Description                        |
|---------------------|------------------------------------|
| `GOOGLE_API_KEY`    | Gemini API key                     |
| `PREFERRED_MODEL`   | Gemini model name                  |
| `SECRET_KEY`        | JWT signing secret                 |
| `ENCRYPTION_KEY`    | Fernet key for email encryption    |
| `EMAIL_TRACKER_DB`  | Path to SQLite email tracker DB    |
| `CREDENTIALS_PATH`  | Gmail OAuth2 credentials.json path |
| `TOKEN_PATH`        | Gmail OAuth2 token.json path       |

---

## Known Limitations

- Free Gemini API tier: 20 requests/day — upgrade for production use
- Email sending requires Gmail OAuth2 setup
- CV extraction works best with text-based PDFs (not scanned images)
- ChromaDB is local — for production use a managed vector DB

---

## License

MIT
