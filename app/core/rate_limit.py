from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.core.security import AuthContext, require_auth


@dataclass
class SlidingWindowRateLimiter:
    window_seconds: int = 60
    _events: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque), init=False)

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        bucket = self._events[key]

        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False

        bucket.append(now)
        return True


limiter = SlidingWindowRateLimiter()


def enforce_rate_limit(
    request: Request,
    auth: AuthContext = Depends(require_auth),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.rate_limit_enabled:
        return

    user_key = f"user:{auth.subject}"
    ip = request.client.host if request.client and request.client.host else "unknown"
    ip_key = f"ip:{ip}"

    user_ok = limiter.allow(user_key, settings.rate_limit_per_user_per_minute)
    ip_ok = limiter.allow(ip_key, settings.rate_limit_per_ip_per_minute)
    if user_ok and ip_ok:
        return

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded",
    )
