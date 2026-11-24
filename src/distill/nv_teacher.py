"""NVIDIA NIM teacher for RAPPORT Stage C distillation.

Implements the project's :class:`~src.distill.teacher.Teacher` protocol
(``__call__(example, prompt) -> Sequence[float]``) on top of the OpenAI SDK
against NVIDIA's integrate endpoint. The teacher returns a probability
distribution over the example's options, which the caller converts to
log-probabilities for the soft-label cache.

The teacher never sees the gold label: it only receives the rendered prompt
(goal, state, options), so its signal is independent supervision rather than a
copy of the annotation.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Any, Sequence

from src.data.core.schema import DecisionExample

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

_SYSTEM = (
    "You are a precise decision engine. Given a goal, a state and a set of "
    "labelled options, judge which option best answers the goal. "
    "Reply with ONLY a JSON array of probabilities, one per option, in the same "
    "order, summing to 1. No explanation, no markdown, no extra keys."
)


class NvidiaTeacherError(RuntimeError):
    code = "NVIDIA_TEACHER_ERROR"


class NvidiaTeacher:
    """Teacher callable backed by NVIDIA NIM (OpenAI-compatible endpoint).

    Parameters
    ----------
    model:
        NIM model id (default ``nvidia/nemotron-3-ultra-550b-a55b``).
    base_url:
        OpenAI-compatible base URL (default NVIDIA integrate endpoint).
    api_key:
        API key. Falls back to the ``NVIDIA_API_KEY`` environment variable and
        then to ``nvapi.key`` in the working directory.
    workers:
        Concurrent requests; the NIM endpoint is rate-limited, so the default is
        deliberately modest.
    rate_per_minute:
        Hard global cap on teacher forwards per minute. The endpoint rejects
        bursts with HTTP 429, so every request passes through one shared
        limiter rather than trusting the worker count.
    retries:
        Attempts per request before the teacher gives up (the caller then sees a
        uniform distribution, never a fabricated one).
    backoff:
        Base seconds for the retry sleep; it grows linearly with the attempt.
    thinking:
        Enable the model's reasoning mode. It improves multi-step judgements but
        is markedly slower and costs more tokens.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        *,
        workers: int = 4,
        rate_per_minute: int = 35,
        retries: int = 4,
        backoff: float = 3.0,
        thinking: bool = False,
        timeout: float = 120.0,
        max_tokens: int = 96,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.workers = max(1, int(workers))
        self.rate_per_minute = max(1, int(rate_per_minute))
        self.retries = max(1, int(retries))
        self.backoff = float(backoff)
        self.thinking = thinking
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or self._read_key_file()
        if not self.api_key:
            raise NvidiaTeacherError(
                "no NVIDIA API key: set NVIDIA_API_KEY, pass api_key=..., or write nvapi.key"
            )
        self._lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = 60.0 / self.rate_per_minute
        self.calls = 0
        self.failures = 0
        self.last_errors: list[str] = []
        self._client = self._build_client()

    def _throttle(self) -> None:
        """Block until the global per-minute budget allows another request."""
        with self._rate_lock:
            now = time.monotonic()
            wait = self._last_call + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_call = now

    @staticmethod
    def _read_key_file() -> str | None:
        path = os.path.join(os.getcwd(), "nvapi.key")
        if os.path.isfile(path):
            try:
                return open(path, encoding="utf-8").read().strip() or None
            except OSError:
                return None
        return None

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise NvidiaTeacherError("the openai package is required for the NIM teacher") from exc
        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    # -- core -----------------------------------------------------------------
    def _ask(self, prompt: str) -> str | None:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if not self.thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        last_error: str = ""
        for attempt in range(self.retries):
            self._throttle()
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    return content
                last_error = "empty content"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                # Rate limits and transient 5xx are retried with backoff; a
                # malformed request is not (it would fail identically).
                if "429" in last_error or "rate" in last_error.lower() or "timeout" in last_error.lower():
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                if "400" in last_error or "Bad Request" in last_error:
                    break
                time.sleep(self.backoff * (attempt + 1))
        with self._lock:
            self.failures += 1
            if len(self.last_errors) < 5:
                self.last_errors.append(last_error)
        return None

    @staticmethod
    def _parse(content: str | None, option_count: int) -> list[float]:
        """Parse the teacher's JSON array into a normalised probability vector."""
        uniform = [1.0 / option_count] * option_count
        if not content:
            return uniform
        text = content.strip()
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return uniform
        try:
            values = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return uniform
        if not isinstance(values, list) or not values:
            return uniform
        cleaned: list[float] = []
        for item in values[:option_count]:
            try:
                value = float(item)
            except (TypeError, ValueError):
                value = 0.0
            if not math.isfinite(value) or value < 0:
                value = 0.0
            cleaned.append(value)
        if len(cleaned) < option_count:
            cleaned.extend([0.0] * (option_count - len(cleaned)))
        total = sum(cleaned)
        if total <= 0:
            return uniform
        return [max(value / total, 1e-6) for value in cleaned]

    def __call__(self, example: DecisionExample, prompt: str) -> Sequence[float]:
        """Return the teacher's probability distribution over the options."""
        probabilities = self._parse(self._ask(prompt), len(example.options))
        with self._lock:
            self.calls += 1
        return [math.log(value) for value in probabilities]

    def batch(self, pairs: Sequence[tuple[DecisionExample, str]]) -> list[list[float]]:
        """Teacher forwards for many examples (used by the cache builder)."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, pairs))

    def _one(self, pair: tuple[DecisionExample, str]) -> list[float]:
        example, prompt = pair
        probabilities = self._parse(self._ask(prompt), len(example.options))
        with self._lock:
            self.calls += 1
        return [math.log(value) for value in probabilities]

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "thinking": self.thinking,
            "rate_per_minute": self.rate_per_minute,
            "last_errors": list(self.last_errors),
        }
