"""Provider-agnostic LLM client with total deadline budget, retries, and daily usage cap.
Supports 'fake' (for tests), 'openai', 'gemini', and generic JSON completion endpoints.
"""
import datetime
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("lic_recommender.llm")


class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when the total deadline budget is exceeded."""
    pass


class LLMCapExceededError(LLMError):
    """Raised when the daily LLM request cap is reached."""
    pass


class LLMClient:
    """Provider-agnostic client with hard timeout budget, retry backoff, and daily quota."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        daily_cap: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        self.enabled = (
            enabled if enabled is not None
            else os.environ.get("ENABLE_LLM", "false").lower() in ("true", "1", "yes")
        )
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "fake")).lower()
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.total_timeout = float(timeout or os.environ.get("LLM_TIMEOUT", "3.0"))
        self.max_tokens = int(max_tokens or os.environ.get("LLM_MAX_TOKENS", "512"))
        self.daily_cap = int(daily_cap or os.environ.get("LLM_DAILY_CAP", "100"))

        # Daily usage counter
        self._current_date = datetime.date.today()
        self._daily_requests = 0

    def _check_and_increment_cap(self):
        today = datetime.date.today()
        if today != self._current_date:
            self._current_date = today
            self._daily_requests = 0

        if self._daily_requests >= self.daily_cap:
            logger.warning("LLM daily cap reached: %d/%d requests", self._daily_requests, self.daily_cap)
            raise LLMCapExceededError(f"Daily request cap of {self.daily_cap} exceeded")

        self._daily_requests += 1

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        timeout_budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute JSON completion with a hard total deadline across all retries."""
        if not self.enabled:
            raise LLMError("LLM layer is disabled (ENABLE_LLM=false)")

        budget = timeout_budget if timeout_budget is not None else self.total_timeout
        start_time = time.monotonic()

        self._check_and_increment_cap()

        last_error = None
        max_attempts = 2

        for attempt in range(max_attempts):
            elapsed = time.monotonic() - start_time
            remaining = budget - elapsed
            if remaining <= 0.1:
                logger.warning("LLM timeout budget exhausted before attempt %d: %.3fs remaining", attempt + 1, remaining)
                raise LLMTimeoutError(f"LLM total deadline of {budget}s exceeded")

            try:
                t0 = time.monotonic()
                raw_response = self._call_provider(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    timeout_seconds=remaining
                )
                latency = time.monotonic() - t0
                parsed = json.loads(raw_response)
                logger.info(
                    "LLM JSON completion successful (attempt=%d, latency=%.3fs, provider=%s)",
                    attempt + 1, latency, self.provider
                )
                return parsed

            except LLMTimeoutError:
                raise
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning("LLM output is not valid JSON (attempt %d): %s", attempt + 1, exc)
                # Schema/JSON errors are not retried with same prompt
                break
            except Exception as exc:
                last_error = exc
                logger.warning("LLM provider error (attempt %d): %s", attempt + 1, exc)
                elapsed = time.monotonic() - start_time
                remaining = budget - elapsed
                if attempt < max_attempts - 1 and remaining > 0.4:
                    backoff = min(0.2, remaining / 2)
                    time.sleep(backoff)
                else:
                    break

        total_elapsed = time.monotonic() - start_time
        if total_elapsed >= budget:
            raise LLMTimeoutError(f"LLM total deadline of {budget}s exceeded (elapsed {total_elapsed:.3f}s)")

        raise LLMError(f"LLM completion failed: {last_error}")

    def _call_provider(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]],
        timeout_seconds: float
    ) -> str:
        """Dispatch call to provider backend."""
        if self.provider == "fake":
            return json.dumps({
                "customer_age": None,
                "child_age": None,
                "term": None,
                "sum_assured": None,
                "gender": None,
                "goal": None,
                "risk_note": None,
                "out_of_scope_flag": False,
                "confidence": 1.0
            })

        if self.provider == "openai":
            if not self.api_key:
                raise LLMError("LLM_API_KEY is required for openai provider")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload: Dict[str, Any] = {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.URLError as e:
                raise LLMError(f"OpenAI connection error: {e}")

        if self.provider == "gemini":
            if not self.api_key:
                raise LLMError("LLM_API_KEY is required for gemini provider")
            model_name = self.model.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload: Dict[str, Any] = {
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": self.max_tokens,
                    "responseMimeType": "application/json"
                }
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise LLMError("Gemini returned no candidates")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts or "text" not in parts[0]:
                        raise LLMError("Gemini response missing text part")
                    return parts[0]["text"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                raise LLMError(f"Gemini API HTTP {e.code}: {err_body}")
            except urllib.error.URLError as e:
                raise LLMError(f"Gemini connection error: {e}")

        raise LLMError(f"Unsupported LLM provider: {self.provider}")

    def chat(
        self,
        messages: list,
        system_prompt: Optional[str] = None,
        timeout_budget: Optional[float] = None
    ) -> str:
        """Execute chat completion with a hard total deadline across all retries."""
        if not self.enabled:
            raise LLMError("LLM layer is disabled (ENABLE_LLM=false)")

        budget = timeout_budget if timeout_budget is not None else max(self.total_timeout, 4.0)
        start_time = time.monotonic()
        self._check_and_increment_cap()

        last_error = None
        max_attempts = 2

        for attempt in range(max_attempts):
            elapsed = time.monotonic() - start_time
            remaining = budget - elapsed
            if remaining <= 0.1:
                raise LLMTimeoutError(f"LLM total deadline of {budget}s exceeded")

            try:
                t0 = time.monotonic()
                response_text = self._call_provider_chat(
                    system_prompt=system_prompt or "",
                    messages=messages,
                    timeout_seconds=remaining
                )
                latency = time.monotonic() - t0
                logger.info(
                    "LLM chat completion successful (attempt=%d, latency=%.3fs, provider=%s)",
                    attempt + 1, latency, self.provider
                )
                return response_text
            except LLMTimeoutError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("LLM provider chat error (attempt %d): %s", attempt + 1, exc)
                elapsed = time.monotonic() - start_time
                remaining = budget - elapsed
                if attempt < max_attempts - 1 and remaining > 0.4:
                    time.sleep(min(0.2, remaining / 2))
                else:
                    break

        raise LLMError(f"LLM chat completion failed: {last_error}")

    def _call_provider_chat(
        self,
        system_prompt: str,
        messages: list,
        timeout_seconds: float
    ) -> str:
        """Dispatch conversational chat call to provider backend."""
        if self.provider == "fake":
            last_msg = messages[-1].get("content", "") if messages else ""
            return f"Thank you for asking about LIC policies. Based on verified brochures, {last_msg[:60]} is covered under our 21 verified plans."

        if self.provider == "openai":
            if not self.api_key:
                raise LLMError("LLM_API_KEY is required for openai provider")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            for m in messages:
                chat_messages.append({
                    "role": m.get("role", "user"),
                    "content": m.get("content", "")
                })
            payload: Dict[str, Any] = {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": min(self.max_tokens * 2, 1024),
                "messages": chat_messages
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.URLError as e:
                raise LLMError(f"OpenAI connection error: {e}")

        if self.provider == "gemini":
            if not self.api_key:
                raise LLMError("LLM_API_KEY is required for gemini provider")
            model_name = self.model.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            gemini_contents = []
            for m in messages:
                role = "user" if m.get("role") == "user" else "model"
                gemini_contents.append({
                    "role": role,
                    "parts": [{"text": m.get("content", "")}]
                })
            payload: Dict[str, Any] = {
                "contents": gemini_contents,
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": min(self.max_tokens * 2, 1024),
                }
            }
            if system_prompt:
                payload["system_instruction"] = {
                    "parts": [{"text": system_prompt}]
                }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise LLMError("Gemini returned no candidates")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts or "text" not in parts[0]:
                        raise LLMError("Gemini response missing text part")
                    return parts[0]["text"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                raise LLMError(f"Gemini API HTTP {e.code}: {err_body}")
            except urllib.error.URLError as e:
                raise LLMError(f"Gemini connection error: {e}")

        raise LLMError(f"Unsupported LLM provider: {self.provider}")


class FakeLLMClient(LLMClient):
    """Deterministic mock client for unit testing timeouts, injections, and edge cases."""

    def __init__(
        self,
        custom_handler: Optional[Callable[[str, str], str]] = None,
        custom_chat_handler: Optional[Callable[[str, list], str]] = None,
        force_timeout: bool = False,
        sleep_delay: float = 0.0,
        daily_cap: int = 100,
        enabled: bool = True
    ):
        super().__init__(provider="fake", enabled=enabled, daily_cap=daily_cap, timeout=3.0)
        self.custom_handler = custom_handler
        self.custom_chat_handler = custom_chat_handler
        self.force_timeout = force_timeout
        self.sleep_delay = sleep_delay

    def _call_provider_chat(
        self,
        system_prompt: str,
        messages: list,
        timeout_seconds: float
    ) -> str:
        if self.sleep_delay > 0:
            time.sleep(self.sleep_delay)

        if self.force_timeout or self.sleep_delay > timeout_seconds:
            raise LLMTimeoutError(f"Simulated timeout: delay {self.sleep_delay}s exceeded {timeout_seconds}s")

        if self.custom_chat_handler:
            return self.custom_chat_handler(system_prompt, messages)

        last_msg = messages[-1].get("content", "") if messages else ""
        return f"Brochure-verified info regarding: {last_msg[:80]}. For instance, LIC's Tech Term (Plan 954) offers pure-risk cover for ages 18-65."

    def _call_provider(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]],
        timeout_seconds: float
    ) -> str:
        if self.sleep_delay > 0:
            time.sleep(self.sleep_delay)

        if self.force_timeout or self.sleep_delay > timeout_seconds:
            raise LLMTimeoutError(f"Simulated timeout: delay {self.sleep_delay}s exceeded {timeout_seconds}s")

        if self.custom_handler:
            return self.custom_handler(system_prompt, user_prompt)

        # Default standard extraction
        return json.dumps({
            "customer_age": None,
            "child_age": None,
            "term": None,
            "sum_assured": None,
            "gender": None,
            "goal": None,
            "risk_note": None,
            "out_of_scope_flag": False,
            "confidence": 1.0
        })
