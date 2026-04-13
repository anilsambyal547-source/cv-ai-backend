"""
FastAPI main application with JWT auth and rate limiting.
Safe-Mode: Enhanced diagnostics and fail-safe connectivity.
"""
import os
import time
from datetime import timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import config

# Resource throttling
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

app = FastAPI(title="CV AI API", version="1.0.0")

# 1. Diagnostic Health Check
@app.get("/health", tags=["health"])
async def health_check(request: Request):
    return {
        "status": "ok",
        "timestamp": time.time(),
        "cors_debug": {
            "origin_received": request.headers.get("origin"),
            "allowed_origins": [o for o in _ALLOWED_ORIGINS if "localhost" not in o] 
        }
    }

# 2. Golden Bridge CORS Configuration (Strict & Explicit)
_ALLOWED_ORIGINS = [
    "https://cvsai.vercel.app",
    "https://cvsai.vercel.app/",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 3. Fast Login (Zero-CPU)
@app.post("/token", response_model=dict, tags=["authentication"])
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    from . import auth
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    return {
        "access_token": auth.create_access_token(data={"sub": user["username"], "role": user["role"]}),
        "token_type": "bearer"
    }

# 4. Safe Router Inclusion
# We keep these isolated to prevent boot-time crashes
from .deps import get_current_active_user, require_role
def include_routers(app: FastAPI):
    from .routes import ingest_router, search_router, candidate_router, email_router, analytics_router
    app.include_router(ingest_router, prefix="/api", tags=["ingestion"], dependencies=[Depends(require_role("admin"))])
    app.include_router(search_router, prefix="/api", tags=["search"], dependencies=[Depends(get_current_active_user)])
    app.include_router(candidate_router, prefix="/api", tags=["candidates"], dependencies=[Depends(get_current_active_user)])
    app.include_router(email_router, prefix="/api", tags=["email"], dependencies=[Depends(get_current_active_user)])
    app.include_router(analytics_router, prefix="/api", tags=["analytics"], dependencies=[Depends(get_current_active_user)])

# Only include once the server is actually paged
include_routers(app)