"""
JWT authentication utilities with role‑based access control.
Optimized for the Render 512MB environment.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from config import config

# Security setup
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Global user database
# We now use plain passwords in the dict and hash them lazily to ensure
# we never miss-match the internal algorithm settings.
_USER_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "plain_password": "secret", 
        "hashed_password": None,
        "role": "admin",
    },
    "recruiter": {
        "username": "recruiter",
        "full_name": "Recruiter User",
        "email": "recruiter@example.com",
        "plain_password": "secret",
        "hashed_password": None,
        "role": "recruiter",
    }
}

def get_user(username: str):
    user = _USER_DB.get(username)
    if not user:
        return None
    # Lazy hash the password on the very first attempt to save startup RAM
    # but ensure it matches the current pwd_context.
    if user["hashed_password"] is None:
        user["hashed_password"] = pwd_context.hash(user["plain_password"])
    return user

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_user(username)
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