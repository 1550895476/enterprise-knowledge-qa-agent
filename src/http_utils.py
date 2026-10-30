from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request
from typing import Any


RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def urlopen_with_retry(
    request: urllib.request.Request,
    timeout: int,
    attempts: int = 3,
) -> Any:
    """Open an HTTP request with bounded retries for transient transport errors."""
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_STATUS or attempt == attempts:
                raise
            exc.close()
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, ConnectionError):
            if attempt == attempts:
                raise
        time.sleep(0.5 * attempt)
    raise RuntimeError("HTTP retry loop ended unexpectedly")
