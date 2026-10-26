"""Disk-cached HTTP client with per-host rate limits and retries.

NCBI E-utilities, Ensembl REST, GTEx, and the Human Protein Atlas are all
reached through this client. Successful JSON bodies are stored under
``data/cache`` so a rerun does not repeat a finished download. Responses that
fail are not cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from rnsp.config import USER_AGENT

logger = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class HttpError(RuntimeError):
    """A request failed after the allowed number of attempts."""

    def __init__(self, message: str, *, status: int | None = None, url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.url = url


class RateLimiter:
    """Sleep so that successive network calls are at least ``min_interval`` apart."""

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last = 0.0

    def wait(self) -> None:
        delay = self.min_interval - (self._clock() - self._last)
        if delay > 0:
            self._sleep(delay)
        self._last = self._clock()


class HttpClient:
    """GET JSON with a SHA-256 disk cache, a rate limit, and exponential backoff."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        min_interval: float = 0.0,
        max_retries: int = 5,
        timeout: tuple[float, float] = (10.0, 90.0),
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.limiter = limiter or RateLimiter(min_interval)
        self._sleep = sleep
        self._rng = rng or random.Random(0)

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Return parsed JSON, using the disk cache when the body was stored."""
        key = self._cache_key("GET", url, params)
        cached = self._read_cache(key)
        if cached is not None:
            return json.loads(cached)

        content = self._request("GET", url, params=params, headers=headers)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise HttpError(
                f"Response from {url} was not JSON",
                url=url,
            ) from exc
        self._write_cache(key, content)
        return payload

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                self._backoff(attempt, url, str(exc))
                continue
            if response.status_code in RETRY_STATUSES:
                last_error = HttpError(
                    f"HTTP {response.status_code} from {response.url}",
                    status=response.status_code,
                    url=str(response.url),
                )
                self._backoff(attempt, url, f"HTTP {response.status_code}")
                continue
            if response.status_code >= 400:
                raise HttpError(
                    f"HTTP {response.status_code} from {response.url}",
                    status=response.status_code,
                    url=str(response.url),
                )
            return response.content
        raise HttpError(f"Gave up on {url}: {last_error}", url=url)

    def _backoff(self, attempt: int, url: str, reason: str) -> None:
        delay = min(60.0, 1.5 * (2**attempt)) + self._rng.random()
        logger.warning(
            "Retrying %s after %s (attempt %s, sleep %.1fs)",
            url,
            reason,
            attempt + 1,
            delay,
        )
        self._sleep(delay)

    def _cache_key(self, method: str, url: str, params: dict[str, Any] | None) -> str:
        raw = json.dumps(
            {"method": method, "url": url, "params": params or {}},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / key[:2] / key

    def _read_cache(self, key: str) -> bytes | None:
        path = self._path_for(key)
        if path.exists():
            return path.read_bytes()
        return None

    def _write_cache(self, key: str, content: bytes) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
