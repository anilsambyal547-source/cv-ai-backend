# CV AI Platform Deployment Guide (Vercel + Render)

This guide provides step-by-step instructions to deploy your CV AI Recruiter Platform as a high-quality portfolio project.

---

## 1. Backend Deployment (Render)

We use **Render** for the backend because it supports **Persistent Disks**, ensuring your indexed CVs and ChromaDB data aren't deleted when the server restarts.

### Steps:
1. **GitHub**: Push your entire repository to a private or public GitHub repo.
2. **Create Blueprint**:
   - Go to your Render Dashboard.
   - Click **New +** > **Blueprint**.
   - Select your repository.
   - Render will automatically detect the `render.yaml` file and set up the Web Service and Persistent Disk.
3. **Environment Variables**:
   - Render will prompt you for the following:
     - `GOOGLE_API_KEY`: Your Gemini API Key.
     - `JWT_SECRET`: A long, random string for secure logins.
     - `ALLOWED_ORIGINS`: Your Vercel frontend URL (e.g., `https://your-app.vercel.app`).
     - `ENCRYPTION_KEY`: A 32-byte base64 string (optional but recommended).

---

## 2. Frontend Deployment (Vercel)

Vercel is the best choice for the Next.js frontend.

### Steps:
1. **GitHub**: Connect your GitHub repo to Vercel.
2. **Configure Framework**: Select **Next.js**.
3. **Environment Variables**:
   - Add `NEXT_PUBLIC_API_URL`: The URL of your Render backend (e.g., `https://cv-ai-backend.onrender.com`).
4. **Deploy**: Click **Deploy**.

---

## 3. Engineering Quality Signals (Portfolio Tips)

When a recruiter looks at this deployment, highlight these features in your README or during an interview:
- **Password Hashing**: We use Bcrypt to ensure user credentials are never stored in plain text.
- **Environment Portability**: The app uses dynamic configuration. The same code works locally and in production just by changing `.env` variables.
- **Stateless/Persistent Hybrid**: You've handled the challenge of "Serverless Frontend" (Vercel) communicating with a "Persistent Backend" (Render).
- **Theme Awareness**: The UI automatically adjusts to system preferences while maintaining a premium "Industrial Glass" aesthetic.

---

## Local Development Reminder
To run locally after these changes:
1. Ensure your `.env` has `NEXT_PUBLIC_API_URL=http://localhost:8000`.
2. Run `npm run dev` in the frontend.
3. Run `python -m uvicorn app.api.main:app --reload` in the backend.
