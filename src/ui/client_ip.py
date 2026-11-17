"""Resolve the client IP of a Streamlit request for per-IP limits."""
from __future__ import annotations

from typing import Mapping, Optional


def resolve_client_ip(
    peer_ip: Optional[str], headers: Optional[Mapping[str, str]], trusted_proxy_hops: int
) -> str:
    """Return the client IP.

    ``X-Forwarded-For`` is only used when the app runs behind ``trusted_proxy_hops``
    reverse proxies; each proxy appends the address it received the request from, so
    the client is the entry that many places from the right. Entries further left are
    written by the client itself and are ignored, as they can be forged.
    """
    if trusted_proxy_hops > 0 and headers:
        forwarded = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for") or ""
        hops = [part.strip() for part in forwarded.split(",") if part.strip()]
        if len(hops) >= trusted_proxy_hops:
            return hops[-trusted_proxy_hops]
    # Streamlit reports None for localhost connections.
    return peer_ip or "127.0.0.1"
