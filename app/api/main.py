"""
FastAPI main application with JWT auth and rate limiting.
Optimized for low-memory environments (Render 512MB).
"""
import os
import logging
from datetime import timedelta

# Force torch to be single-threaded and memory-efficient
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="CV AI API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 1. Health check first
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

# 2. Hardened CORS Configuration
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
# Clean origins: strip spaces and trailing slashes to prevent handshake failures
_ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in raw_origins.split(",") if o.strip()]

# Log allowed origins for debugging
print(f"DEBUG: CORS origins configured: {_ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Login Endpoint
@app.post("/token", response_model=dict, tags=["authentication"])
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    from . import auth
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

# 4. Lazy Router Inclusion
from .deps import get_current_active_user, require_role
def include_routers(app: FastAPI):
    from .routes import (
        ingest_router,
        search_router,
        candidate_router,
        email_router,
        analytics_router
    )
    
    app.include_router(ingest_router, prefix="/api", tags=["ingestion"], dependencies=[Depends(require_role("admin"))])
    app.include_router(search_router, prefix="/api", tags=["search"], dependencies=[Depends(get_current_active_user)])
    app.include_router(candidate_router, prefix="/api", tags=["candidates"], dependencies=[Depends(get_current_active_user)])
    app.include_router(email_router, prefix="/api", tags=["email"], dependencies=[Depends(get_current_active_user)])
    app.include_router(analytics_router, prefix="/api", tags=["analytics"], dependencies=[Depends(get_current_active_user)])

include_routers(app)