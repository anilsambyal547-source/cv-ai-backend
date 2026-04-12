"""
JWT authentication utilities with role‑based access control.
Bypasses bcrypt for simplicity – uses hardcoded passwords.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt

from passlib.context import CryptContext
from config import config

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User database with hashed passwords
# Note: In a real app these would be in a SQL DB.
# 'secret' hashed: $2b$12$6qV9W69396796969696969O
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
        "role": "admin",
    },
    "recruiter": {
        "username": "recruiter",
        "full_name": "Recruiter User",
        "email": "recruiter@example.com",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
        "role": "recruiter",
    }
}

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)
    return encoded_jwt