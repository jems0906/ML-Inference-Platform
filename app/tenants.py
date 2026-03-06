import re
import secrets
from typing import Dict

# tenant_id must be 3-32 lowercase alphanumeric/hyphen/underscore characters
_TENANT_ID_RE = re.compile(r"^[a-z0-9_-]{3,32}$")

# In-memory store — replace with a database in production
TENANTS: Dict[str, Dict] = {
    "tenant1": {"api_key": "key1", "rate_limit": 10, "plan": "standard"},
    "tenant2": {"api_key": "key2", "rate_limit": 5, "plan": "free"},
}


def get_tenant_by_api_key(api_key: str) -> str | None:
    for tenant_id, data in TENANTS.items():
        if data["api_key"] == api_key:
            return tenant_id
    return None


def get_tenant_rate_limit(tenant_id: str) -> int:
    return TENANTS.get(tenant_id, {}).get("rate_limit", 0)


def list_tenants() -> list:
    return [
        {"tenant_id": tid, "plan": data["plan"], "rate_limit": data["rate_limit"]}
        for tid, data in TENANTS.items()
    ]


def create_tenant(tenant_id: str, rate_limit: int = 10, plan: str = "standard") -> dict | None:
    if not _TENANT_ID_RE.match(tenant_id):
        raise ValueError(f"tenant_id must match pattern {_TENANT_ID_RE.pattern}")
    if tenant_id in TENANTS:
        return None
    api_key = secrets.token_hex(16)
    TENANTS[tenant_id] = {"api_key": api_key, "rate_limit": rate_limit, "plan": plan}
    return {"tenant_id": tenant_id, "api_key": api_key, "rate_limit": rate_limit, "plan": plan}