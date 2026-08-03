# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry policy for remote model calls.

Ported from SimFoundry's `simfoundry/models/vlm.py`; keep the two in sync.
Rate limits get a larger attempt budget than other errors; client errors that
cannot succeed on retry fail immediately.
"""

import os
import random
import re
import time

RETRY_BASE_DELAY_S = float(os.environ.get("SIMFOUNDRY_REMOTE_RETRY_BASE_S", 2.0))
RETRY_MAX_DELAY_S = float(os.environ.get("SIMFOUNDRY_REMOTE_RETRY_MAX_S", 60.0))
RATE_LIMIT_MAX_ATTEMPTS = int(os.environ.get("SIMFOUNDRY_REMOTE_RATE_LIMIT_ATTEMPTS", 8))

_RATE_LIMIT_MARKERS = (
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "rate limit",
    "ratelimit",
    "quota exceeded",
    "too many requests",
)
_NON_RETRYABLE_MARKERS = (
    "400",
    "401",
    "403",
    "404",
    "invalid_argument",
    "permission_denied",
    "unauthenticated",
    "not_found",
    "api key not valid",
)


class RemoteCallFailed(RuntimeError):
    """A remote model call exhausted its retries."""


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when an exception looks like provider-side throttling."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def is_non_retryable_error(exc: BaseException) -> bool:
    """True for client errors that will not succeed on retry."""
    if is_rate_limit_error(exc):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _NON_RETRYABLE_MARKERS)


def retry_delay_from_error(exc: BaseException):
    """Honor a server-provided retry delay when the SDK surfaces one."""
    for attr in ("retry_delay", "retry_after"):
        value = getattr(exc, attr, None)
        seconds = getattr(value, "seconds", value)
        try:
            if seconds is not None and float(seconds) > 0:
                return float(seconds)
        except (TypeError, ValueError):
            pass
    match = re.search(r"retry[_ -]?(?:delay|after)\D{0,12}?(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def backoff_sleep_s(attempt: int, exc: BaseException = None) -> float:
    """Exponential backoff with full jitter; `attempt` is 0-based."""
    if exc is not None:
        server_delay = retry_delay_from_error(exc)
        if server_delay is not None:
            return min(server_delay, RETRY_MAX_DELAY_S)
    ceiling = min(RETRY_BASE_DELAY_S * (2 ** attempt), RETRY_MAX_DELAY_S)
    return random.uniform(0.0, ceiling)


def handle_remote_exception(exc, *, attempt, n_retries, provider, model, sleep_fn=time.sleep):
    """Sleep and report, or re-raise. Returns the effective attempt budget."""
    if is_non_retryable_error(exc):
        raise RemoteCallFailed(
            f"{provider} [{model}] failed with a non-retryable error: {exc}"
        ) from exc

    rate_limited = is_rate_limit_error(exc)
    budget = max(n_retries, RATE_LIMIT_MAX_ATTEMPTS) if rate_limited else n_retries
    if attempt + 1 >= budget:
        return budget

    delay = backoff_sleep_s(attempt, exc)
    kind = "rate limited (429)" if rate_limited else "error"
    print(
        f"{provider} [{model}] {kind} on attempt {attempt + 1}/{budget}; "
        f"retrying in {delay:.1f}s: {exc}",
        flush=True,
    )
    sleep_fn(delay)
    return budget
