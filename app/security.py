"""Optional shared-secret auth.

Set ``API_KEYS`` (comma separated) to require a key; leave it unset and the
service accepts every request, which is the right default when it sits on a
private network. The key may arrive in the configured header or as a
``Bearer`` token.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status

from .config import get_settings


async def require_api_key(
    request: Request,
    authorization: str | None = Header(None),
) -> None:
    settings = get_settings()
    if not settings.api_keys:
        return

    presented = request.headers.get(settings.api_key_header)
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    if not presented:
        presented = request.query_params.get("api_key")

    if not presented or not any(hmac.compare_digest(presented, key) for key in settings.api_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing or invalid API key (send it in {settings.api_key_header} or as a Bearer token)",
            headers={"WWW-Authenticate": "Bearer"},
        )
