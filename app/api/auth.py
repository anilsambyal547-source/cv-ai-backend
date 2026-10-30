"""
JWT authentication utilities with role‑based access control.
Ultra-Lightweight Mode for Render Free Tier (Zero Hashing).
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt
from config import config

# Universal User DB (Zero hashing for maximum responsiveness on 512MB RAM)
_USER_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "password": "secret", 
        "role": "admin",
    },
    "recruiter": {
        "username": "recruiter",
        "full_name": "Recruiter User",
        "email": "recruiter@example.com",
        "password": "secret",
        "role": "recruiter",
    }
}

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    # Case-insensitive username + password — works for anyone
    user = _USER_DB.get(username.strip().lower())
    if user and user["password"] == password.strip().lower():
        return user
    return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)