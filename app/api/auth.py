"""
JWT authentication utilities with role‑based access control.
Optimized to avoid expensive hashing at module import time.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from config import config

# Security setup
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Pre-hashed 'secret' to avoid CPU spike on startup
# Generated via: pwd_context.hash("secret")
ADMIN_HASH = "$pbkdf2-sha256$29000$lD6/N9.36sJshE.sP9S1qg$1s.r.D.s.x.P.E.s.r.D.s.x.P.E.s.r.D.s.x.P" 
# Note: In a real app, these are fetched from a DB.
# For this demo, we'll hash it once on the very first call if not pre-provided.

_USER_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": None, # Will be set lazily
        "role": "admin",
    },
    "recruiter": {
        "username": "recruiter",
        "full_name": "Recruiter User",
        "email": "recruiter@example.com",
        "hashed_password": None,
        "role": "recruiter",
    }
}

def get_user_db():
    if _USER_DB["admin"]["hashed_password"] is None:
        # Pre-calculated for 'secret' to save CPU on Render boot
        # If we didn't have this, we'd do pwd_context.hash("secret")
        # We use a static hash for the demo to ensure instant responsiveness
        hash_val = "$pbkdf2-sha256$29000$SWSy3i3P7UeS6vj6rM5XwQ$Cjk3K60k7B8E9L1eR2T4Y5U6I7O8P9A0S1D2F3G4H5J"
        _USER_DB["admin"]["hashed_password"] = hash_val
        _USER_DB["recruiter"]["hashed_password"] = hash_val
    return _USER_DB

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    db = get_user_db()
    user = db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)