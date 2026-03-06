import logging
import time
from collections import defaultdict

import redis

from app.config import settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60

try:
    redis_client = redis.from_url(settings.redis_url, socket_connect_timeout=2)
    redis_client.ping()
except Exception:
    logger.warning("Redis unavailable, falling back to in-memory rate limiting")
    redis_client = None

# Sliding-window fallback: {tenant_id: [request_timestamps]}
_memory_windows: dict = defaultdict(list)


def is_rate_limited(tenant_id: str) -> bool:
    # Lazy import to avoid circular dependency
    from app.tenants import get_tenant_rate_limit

    rate_limit = get_tenant_rate_limit(tenant_id)
    key = f"rate_limit:{tenant_id}"

    if redis_client:
        try:
            current = redis_client.incr(key)
            if current == 1:
                redis_client.expire(key, WINDOW_SECONDS)
            return current > rate_limit
        except Exception:
            pass  # fall through to in-memory

    # Sliding-window in-memory fallback
    now = time.time()
    window = [t for t in _memory_windows[tenant_id] if now - t < WINDOW_SECONDS]
    window.append(now)
    _memory_windows[tenant_id] = window
    return len(window) > rate_limit