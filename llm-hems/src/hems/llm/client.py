"""Provider-agnostic LLM wrapper.

Responsibility (BUILD_SPEC.md §14): wraps the chosen provider SDK behind one
interface so it never leaks into agent logic. Reads API keys from the
environment (`.env`, gitignored -- see `llm-hems/.env`).

Two backends are supported, selected by `provider` ("matilda" | "gemini",
normally set via `config/experiment.yaml`'s `llm.provider`):

- **matilda**: Maincode's Matilda API (`matilda-client` on PyPI), confirmed
  live 2026-09-06 against `https://matilda.maincode.com/api`. Matilda owns
  model/provider routing internally -- its `chat.create_object` call exposes
  no `model`, `temperature`, or `seed` parameter, so there is no model id to
  record beyond a fixed placeholder string (see README.md "Reproducibility:
  the LLM agent"). Async-only (httpx-based); this wrapper runs each call to
  completion with `asyncio.run` so agent code (synchronous per the
  `DecisionAgent` protocol) never touches asyncio directly.
- **gemini**: Google's `google-genai` SDK against the Gemini API
  (`GEMINI_API_KEY`). Model id is configurable (`llm.gemini.model` in
  experiment.yaml); default `gemini-3.6-flash`, confirmed live 2026-09-06 --
  `gemini-2.5-flash` has been retired (404s, server message points to
  `gemini-3.6-flash`), and the newer `gemini-3.8-flash` was 503ing under high
  demand at confirmation time, so 3.6 is the safer default for a reliability
  battery. Structured output is requested via `response_json_schema`
  (pydantic `model_json_schema()`), matched against each appliance's
  response schema in `llm/schemas.py`. Free-tier calls are capped at 15
  req/min by default (`llm.gemini.rate_limit_rpm`) with a simple sliding-
  window throttle before each call.

Exposes one operation per client: `complete_json(prompt, schema) ->
(parsed, was_cached)`, with a disk-backed cache keyed on `(provider,
prompt)` (§15 cost guard -- identical calls to the same provider are never
re-sent, and switching providers can't collide on a stale cache entry) and a
bounded retry with exponential backoff on transient API errors or schema-
validation failure (§1 pydantic contract). `LLMClient.model_id` /
`.sdk_version` let callers record which exact backend produced a run's
results (BUILD_SPEC.md design principle §3 -- every run writes its resolved
config).
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

PROVIDERS = ("matilda", "gemini")

DEFAULT_MATILDA_BASE_URL = "https://matilda.maincode.com/api"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# gemini-3.6-flash (the flagship Flash model, confirmed live 2026-09-06 --
# see the module docstring) has a free-tier quota of only 20 requests/DAY
# per project+model, which a ~26-call reliability battery exhausts on its
# own; gemini-3.5-flash-lite is the same Flash family at a lighter tier,
# confirmed live the same day with no such daily ceiling hit, so it's the
# workable default for this battery. Re-check both models' quotas before
# assuming this holds on a later run.
# The task brief assumed a 15 req/min free tier; the API's own 429 response
# for this model/key on 2026-09-06 reported an actual free-tier quota of 5
# req/min ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier" = 5) --
# throttle to the observed real limit, not the assumed one. Configurable via
# `llm.gemini.rate_limit_rpm` in experiment.yaml.
DEFAULT_GEMINI_RATE_LIMIT_RPM = 5
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 2048  # generous: gemini-3.x "thinking" eats hidden budget
                                          # first before the JSON body -- 300 truncated the
                                          # response to an empty preamble in testing.


class LLMError(RuntimeError):
    """Raised when the provider call fails after all retries are exhausted."""


class _Backend(abc.ABC):
    """One provider's structured-output call, isolated from caching/retry."""

    name: str

    @abc.abstractmethod
    def call(self, prompt: str, schema: Type[BaseModel]) -> dict: ...

    @property
    @abc.abstractmethod
    def model_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def sdk_version(self) -> str: ...


class _MatildaBackend(_Backend):
    name = "matilda"

    def __init__(self, api_key: str | None, base_url: str, timeout_s: float):
        resolved_key = api_key or os.environ.get("MAINCODE_API_KEY")
        if not resolved_key:
            raise LLMError(
                "MAINCODE_API_KEY is not set -- put it in llm-hems/.env "
                "(gitignored) as MAINCODE_API_KEY=..."
            )
        self._api_key = resolved_key
        self._base_url = base_url
        self._timeout_s = timeout_s

    def call(self, prompt: str, schema: Type[BaseModel]) -> dict:
        return asyncio.run(self._call(prompt, schema))

    async def _call(self, prompt: str, schema: Type[BaseModel]) -> dict:
        import matilda_client as mc

        client = mc.MatildaClient(token=self._api_key, base_url=self._base_url)
        try:
            resp = await asyncio.wait_for(
                client.chat.create_object(schema.model_json_schema(), input=prompt),
                timeout=self._timeout_s,
            )
            return resp.object
        finally:
            await client.aclose()

    @property
    def model_id(self) -> str:
        # Matilda owns model/provider routing internally -- chat.create_object
        # exposes no `model` parameter, so there is no id to report (see
        # README "Reproducibility: the LLM agent").
        return "matilda (routing internal, unexposed by API)"

    @property
    def sdk_version(self) -> str:
        try:
            return f"matilda-client=={importlib.metadata.version('matilda-client')}"
        except importlib.metadata.PackageNotFoundError:
            return "matilda-client==unknown"


class _GeminiBackend(_Backend):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        rate_limit_rpm: int,
        max_output_tokens: int,
    ):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise LLMError(
                "GEMINI_API_KEY is not set -- put it in llm-hems/.env "
                "(gitignored) as GEMINI_API_KEY=..."
            )
        self._api_key = resolved_key
        self._model = model
        self._rate_limit_rpm = rate_limit_rpm
        self._max_output_tokens = max_output_tokens
        self._call_times: list[float] = []

    def _throttle(self) -> None:
        """Free-tier guard: keep to `rate_limit_rpm` calls per rolling 60s window."""
        now = time.monotonic()
        self._call_times = [t for t in self._call_times if now - t < 60.0]
        if len(self._call_times) >= self._rate_limit_rpm:
            sleep_for = 60.0 - (now - self._call_times[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self._call_times = [t for t in self._call_times if now - t < 60.0]
        self._call_times.append(now)

    def call(self, prompt: str, schema: Type[BaseModel]) -> dict:
        from google import genai
        from google.genai import types

        self._throttle()
        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
                max_output_tokens=self._max_output_tokens,
            ),
        )
        if not response.text:
            raise LLMError(
                f"gemini returned no text (finish_reason="
                f"{response.candidates[0].finish_reason if response.candidates else None}) "
                f"-- likely truncated by max_output_tokens"
            )
        return json.loads(response.text)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def sdk_version(self) -> str:
        try:
            return f"google-genai=={importlib.metadata.version('google-genai')}"
        except importlib.metadata.PackageNotFoundError:
            return "google-genai==unknown"


class LLMClient:
    """Provider-agnostic structured-output client (BUILD_SPEC.md §14).

    Selects a backend (`matilda` | `gemini`) and wraps it with a disk cache
    keyed on `(provider, prompt)` and a bounded retry with exponential
    backoff on transient errors or schema-validation failure. Agent code
    (`agents/llm_agent.py`) only ever calls `complete_json`; it never touches
    a provider SDK directly.
    """

    def __init__(
        self,
        provider: str = "matilda",
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        max_retries: int = 2,
        base_url: str = DEFAULT_MATILDA_BASE_URL,
        timeout_s: float = 60.0,
        model: str = DEFAULT_GEMINI_MODEL,
        rate_limit_rpm: int = DEFAULT_GEMINI_RATE_LIMIT_RPM,
        max_output_tokens: int = DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    ):
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider '{provider}' -- choose one of {PROVIDERS}")
        self.provider = provider
        self._backend: _Backend
        if provider == "matilda":
            self._backend = _MatildaBackend(api_key, base_url, timeout_s)
        else:
            self._backend = _GeminiBackend(api_key, model, rate_limit_rpm, max_output_tokens)
        self._max_retries = max_retries
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self._cache_path and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @property
    def sdk_version(self) -> str:
        return self._backend.sdk_version

    def complete_json(self, prompt: str, schema: Type[T]) -> tuple[T, bool]:
        """Returns `(parsed_response, was_cached)`.

        Caches on `(provider, prompt)`, so any change to the context
        (indoor temp, clock, deadline, ...) that a template renders into the
        prompt naturally busts the cache, and switching providers for the
        same prompt text never hits a stale entry from the other provider.
        """
        key = hashlib.sha256(f"{self.provider}:{prompt}".encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return schema.model_validate(cached), True

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._backend.call(prompt, schema)
                parsed = schema.model_validate(raw)
                self._cache[key] = parsed.model_dump()
                self._flush_cache()
                return parsed, False
            except Exception as exc:  # noqa: BLE001 -- provider raises opaque SDK error types
                last_error = exc
                if attempt < self._max_retries:
                    # Rate-limit errors (e.g. Gemini free-tier 429s) need a
                    # longer backoff than transient network errors -- the
                    # provider's own retryDelay is typically 15-20s, well
                    # past a plain exponential-backoff ceiling of 8s.
                    is_rate_limited = "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)
                    sleep_s = max(20, 2**attempt) if is_rate_limited else min(2**attempt, 8)
                    time.sleep(sleep_s)
        raise LLMError(
            f"LLM call failed after {self._max_retries + 1} attempt(s) "
            f"[provider={self.provider}]: {last_error}"
        ) from last_error

    def _flush_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2))


if __name__ == "__main__":
    # Minimal connectivity smoke test: confirms auth + structured output work
    # end to end for either provider. Run with:
    #   python -m hems.llm.client [matilda|gemini]
    import sys

    from hems.llm.schemas import HVACWTPResponse

    provider_arg = sys.argv[1] if len(sys.argv) > 1 else "matilda"
    test_client = LLMClient(provider=provider_arg)
    result, was_cached = test_client.complete_json(
        "CURRENT STATE: indoor temperature is 27.0 C. comfort band is 22.0 C "
        "to 26.0 C. CPP event active right now: no. Household thermal "
        "sensitivity is moderate. Give your willingness-to-pay in dollars "
        "for running the air conditioner this half-hour, a one-line reason, "
        "and echo back the indoor temp, band bounds, and CPP flag exactly "
        "as given above. Do not compute cost or do arithmetic.",
        HVACWTPResponse,
    )
    print(
        f"provider={test_client.provider} model_id={test_client.model_id} "
        f"sdk_version={test_client.sdk_version} cached={was_cached} "
        f"wtp={result.wtp} reason={result.reason!r}"
    )
