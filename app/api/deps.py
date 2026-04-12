"""
Dependencies for FastAPI routes.
Includes get_current_user (JWT validation) and role‑based access control.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional

from . import auth
from config import config

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")  # retrieve role from token
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = auth.fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    # Add role to returned user dict
    user["role"] = role
    return user

async def get_current_active_user(current_user = Depends(get_current_user)):
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_role(required_role: str):
    """
    Dependency to enforce a specific role.
    Usage: `Depends(require_role("admin"))`
    """
    def role_checker(current_user = Depends(get_current_active_user)):
        if current_user.get("role") != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return current_user
    return role_checker