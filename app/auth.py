"""
MCP Auth Starter — JWT verification.
"""

import logging
from typing import Dict

import jwt
from jwt import PyJWTError

from config import SECRET_KEY, ALGORITHM, MCP_RESOURCE_URI

logger = logging.getLogger("mcp-auth-starter")


def _audience_ok(payload: dict) -> bool:
    """RFC 8707. A token carrying "aud" must name this server; one without the
    claim predates audience binding and is left unchecked rather than rejected.

    Checked here rather than by passing audience= to jwt.decode, because PyJWT
    rejects a missing "aud" outright once an audience is supplied.
    """
    aud = payload.get("aud")
    if aud is None:
        return True
    if isinstance(aud, str):
        return aud == MCP_RESOURCE_URI
    return MCP_RESOURCE_URI in aud


async def verify_token(token: str) -> Dict:
    """
    Verify a JWT access token and extract the user's identity.

    Token payload:
    {
        "sub": "username",
        "teams": ["admins", "beta"],
        "exp": timestamp
    }

    Returns: {"username": str, "teams": List[str]}
    Raises: ValueError if token invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM],
                             options={"verify_aud": False})
        if not _audience_ok(payload):
            logger.warning("Token audience does not name this server")
            raise ValueError("Invalid token: audience mismatch")

        username = payload.get("sub")
        teams = payload.get("teams", [])

        if not username:
            logger.warning("Token missing 'sub' claim")
            raise ValueError("Invalid token: missing username")

        if not isinstance(teams, list):
            logger.warning(f"Token teams not a list: {teams}")
            raise ValueError("Invalid token: teams must be array")

        logger.debug(f"Token verified for {username}, teams: {teams}")
        return {"username": username, "teams": teams}

    except PyJWTError as e:
        logger.warning(f"JWT decode error: {str(e)}")
        raise ValueError(f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise ValueError(f"Token verification failed: {str(e)}")
