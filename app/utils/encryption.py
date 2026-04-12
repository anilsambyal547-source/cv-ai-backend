"""
Symmetric encryption utilities using Fernet.
"""
from cryptography.fernet import Fernet
from config import config

def get_cipher():
    key = config.ENCRYPTION_KEY
    if not key:
        raise RuntimeError("ENCRYPTION_KEY not set in environment")
    return Fernet(key.encode())

def encrypt(text: str) -> str:
    """Encrypt a string, return base64 encrypted string."""
    if not text:
        return ""
    cipher = get_cipher()
    return cipher.encrypt(text.encode()).decode()

def decrypt(token: str) -> str:
    """Decrypt a base64 encrypted string."""
    if not token:
        return ""
    cipher = get_cipher()
    return cipher.decrypt(token.encode()).decode()