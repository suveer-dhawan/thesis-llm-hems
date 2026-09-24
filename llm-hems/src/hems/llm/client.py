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
  window throttle before each call. `temperature` and `seed` (BUILD_SPEC.md
  §2.3, `config/experiment.yaml`'s `llm.temperature` / `llm.seed`, default
  0.0 / 42) are passed into `GenerateContentConfig` on every call --
  `probe_wtp_price_v3.py` found temperature=0 alone does NOT make two
  identical prompts return an identical wtp; the seed is required too.

Exposes one operation per client: `complete_json(prompt, schema) ->
(parsed, was_cached)`, with a disk-backed cache keyed on `(provider, model,
temperature, seed, prompt)` (§15 cost guard -- an identical call is never
re-sent, and a different provider, model, or seed can't collide on a stale
entry; Phase 3's multi-seed spread depends on the seed being in the key).
A `.jsonl` cache path is append-only and shared by every client in the
process that names the same file, so re-runs cost zero live calls; a legacy
`.json` path is still read and rewritten whole. `LLMClient.stats` counts
live calls vs cache hits. The Gemini rate-limit throttle is shared across
all clients for the same model, so building one client per seed can't
exceed the per-minute quota. Transient API errors and schema-validation
failures (§1 pydantic contract) get a bounded retry with exponential
backoff. `LLMClient.model_id` /
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
# Reproducibility (BUILD_SPEC.md §2.3, config/experiment.yaml's `llm.temperature`
# / `llm.seed`): probe_wtp_price_v3.py found temperature=0 ALONE is not enough
# for two identical Gemini prompts to return an identical wtp -- the `seed`
# parameter is also required for the determinism check to pass. Both are
# threaded into GenerateContentConfig below.
DEFAULT_GEMINI_TEMPERATURE = 0.0
DEFAULT_GEMINI_SEED = 42


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

    def cache_identity(self) -> dict:
        """Everything besides the prompt that can change the response."""
        return {"provider": self.name}


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
        temperature: float,
        seed: int,
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
        self._temperature = temperature
        self._seed = seed

    # Shared by every backend instance for the same model: the quota is per
    # project+model, not per client object.
    _call_times_by_model: dict[str, list[float]] = {}

    def _throttle(self) -> None:
        """Free-tier guard: keep to `rate_limit_rpm` calls per rolling 60s window."""
        times = self._call_times_by_model.setdefault(self._model, [])
        now = time.monotonic()
        times[:] = [t for t in times if now - t < 60.0]
        if len(times) >= self._rate_limit_rpm:
            sleep_for = 60.0 - (now - times[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            times[:] = [t for t in times if now - t < 60.0]
        times.append(now)

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
                temperature=self._temperature,
                seed=self._seed,
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

    def cache_identity(self) -> dict:
        return {
            "provider": self.name,
            "model": self._model,
            "temperature": self._temperature,
            "seed": self._seed,
        }

    @property
    def sdk_version(self) -> str:
        try:
            return f"google-genai=={importlib.metadata.version('google-genai')}"
        except importlib.metadata.PackageNotFoundError:
            return "google-genai==unknown"


class _DiskCache:
    """key -> response dict, persisted to disk; one shared instance per path.

    `.jsonl`: append-only, one `{"key", "value"}` object per line -- cheap to
    write after every live call and safe to grow to tens of thousands of
    entries. Anything else: the legacy single JSON dict, rewritten whole.
    """

    _instances: dict[Path, "_DiskCache"] = {}

    @classmethod
    def for_path(cls, path: Path) -> "_DiskCache":
        resolved = path.resolve()
        if resolved not in cls._instances:
            cls._instances[resolved] = cls(resolved)
        return cls._instances[resolved]

    def __init__(self, path: Path):
        self.path = path
        self._jsonl = path.suffix == ".jsonl"
        self.data: dict[str, dict] = {}
        if path.exists():
            if self._jsonl:
                for line in path.read_text().splitlines():
                    if line.strip():
                        entry = json.loads(line)
                        self.data[entry["key"]] = entry["value"]
            else:
                self.data = json.loads(path.read_text())

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._jsonl:
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")
        else:
            self.path.write_text(json.dumps(self.data, indent=2))


class LLMClient:
    """Provider-agnostic structured-output client (BUILD_SPEC.md §14).

    Selects a backend (`matilda` | `gemini`) and wraps it with a disk cache
    keyed on `(provider, model, temperature, seed, prompt)` and a bounded retry with exponential
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
        temperature: float = DEFAULT_GEMINI_TEMPERATURE,
        seed: int = DEFAULT_GEMINI_SEED,
    ):
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider '{provider}' -- choose one of {PROVIDERS}")
        self.provider = provider
        self._backend: _Backend
        if provider == "matilda":
            self._backend = _MatildaBackend(api_key, base_url, timeout_s)
        else:
            self._backend = _GeminiBackend(
                api_key, model, rate_limit_rpm, max_output_tokens, temperature, seed
            )
        self._max_retries = max_retries
        self._disk = _DiskCache.for_path(Path(cache_path)) if cache_path else None
        self._cache: dict[str, dict] = self._disk.data if self._disk else {}
        self.stats = {"live_calls": 0, "cache_hits": 0}

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @property
    def sdk_version(self) -> str:
        return self._backend.sdk_version

    def cache_key(self, prompt: str) -> str:
        """Exact-prompt key, scoped to everything else that changes the answer.

        Any change to the context (indoor temp, clock, deadline, ...) that a
        template renders into the prompt busts the cache; so does a different
        provider, model, temperature, or seed.
        """
        identity = {**self._backend.cache_identity(), "prompt": prompt}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()

    def is_cached(self, prompt: str) -> bool:
        return self.cache_key(prompt) in self._cache

    def complete_json(self, prompt: str, schema: Type[T]) -> tuple[T, bool]:
        """Returns `(parsed_response, was_cached)`."""
        key = self.cache_key(prompt)
        cached = self._cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return schema.model_validate(cached), True

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                self.stats["live_calls"] += 1
                raw = self._backend.call(prompt, schema)
                parsed = schema.model_validate(raw)
                if self._disk is not None:
                    self._disk.put(key, parsed.model_dump())
                else:
                    self._cache[key] = parsed.model_dump()
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


def build_llm_client(
    provider: str, llm_cfg: dict, cache_path: str | Path | None, seed: int | None = None
) -> LLMClient:
    """An `LLMClient` for `provider` from experiment.yaml's `llm:` block.

    Pulls provider-specific settings (model, rate limit, base_url, ...) out of
    `llm_cfg`. `seed` overrides `llm.seed` (Phase 3's multi-seed spread).
    """
    kwargs: dict = {"provider": provider, "cache_path": cache_path}
    provider_cfg = llm_cfg.get(provider) or {}
    if provider == "gemini":
        for key in ("model", "rate_limit_rpm", "max_output_tokens"):
            if provider_cfg.get(key) is not None:
                kwargs[key] = provider_cfg[key]
        # Top-level (provider-agnostic) reproducibility knobs -- only Gemini's
        # API exposes them; Matilda's does not.
        if llm_cfg.get("temperature") is not None:
            kwargs["temperature"] = llm_cfg["temperature"]
        resolved_seed = seed if seed is not None else llm_cfg.get("seed")
        if resolved_seed is not None:
            kwargs["seed"] = resolved_seed
    elif provider == "matilda":
        if provider_cfg.get("base_url"):
            kwargs["base_url"] = provider_cfg["base_url"]
    return LLMClient(**kwargs)


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
