from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    subject: str
    auth_type: str


def _api_keys(settings: Settings) -> set[str]:
    keys = [item.strip() for item in settings.api_keys.split(",") if item.strip()]
    return set(keys)


def _decode_jwt(token: str, settings: Settings) -> dict:
    try:  # pragma: no cover - optional dependency
        import jwt
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT support is not installed (install PyJWT)",
        ) from exc

    options = {"require": ["exp", "sub"]}
    kwargs = {}
    if settings.jwt_audience:
        kwargs["audience"] = settings.jwt_audience
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm], options=options, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid JWT: {exc}") from exc

    exp_ts = payload.get("exp")
    if exp_ts is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT is missing exp claim")
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    if int(exp_ts) <= now_ts:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT token is expired")
    return payload


def require_auth(
    settings: Settings = Depends(get_settings),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> AuthContext:
    if not settings.auth_enabled:
        return AuthContext(subject="anonymous", auth_type="disabled")

    valid_api_keys = _api_keys(settings)
    if x_api_key and x_api_key in valid_api_keys:
        return AuthContext(subject=f"api_key:{x_api_key[:6]}", auth_type="api_key")

    if credentials is not None:
        payload = _decode_jwt(credentials.credentials, settings)
        return AuthContext(subject=str(payload.get("sub")), auth_type="jwt")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing credentials: provide Bearer JWT or X-API-Key",
    )
