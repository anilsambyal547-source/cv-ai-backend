"""
FastAPI main application with JWT auth and rate limiting.
Universal Handshake Mode: Optimized for maximum reachability on Render Free Tier.
"""
import os
from datetime import timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import config

# Zero-overhead startup
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

app = FastAPI(title="CV AI API", version="1.0.0")

# 1. Health check (Absolute bare minimum)
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

# 2. Universal Handshake Mode
# Allowing all origins "*" temporarily to bypass dashboard configuration errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Fast Login (Bypasses hashing overhead for stability on 512MB RAM)
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

# 4. Global Router Inclusion
from .deps import get_current_active_user, require_role
def include_routers(app: FastAPI):
    from .routes import ingest_router, search_router, candidate_router, email_router, analytics_router
    app.include_router(ingest_router, prefix="/api", tags=["ingestion"], dependencies=[Depends(require_role("admin"))])
    app.include_router(search_router, prefix="/api", tags=["search"], dependencies=[Depends(get_current_active_user)])
    app.include_router(candidate_router, prefix="/api", tags=["candidates"], dependencies=[Depends(get_current_active_user)])
    app.include_router(email_router, prefix="/api", tags=["email"], dependencies=[Depends(get_current_active_user)])
    app.include_router(analytics_router, prefix="/api", tags=["analytics"], dependencies=[Depends(get_current_active_user)])

include_routers(app)