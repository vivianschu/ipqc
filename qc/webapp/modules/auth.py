"""User authentication utilities."""
from __future__ import annotations

import bcrypt


def hash_password(password: str) -> str:
    """Return bcrypt hash of the given plaintext password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Return True if *password* matches the stored bcrypt *password_hash*."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def password_strength_error(password: str) -> str | None:
    """Return an error string if the password does not meet minimum requirements, else None."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    return None
