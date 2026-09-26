"""Provider dispatch + caching for the provider-agnostic LLM client.

No live API calls -- exercises `LLMClient`'s provider selection, error
handling, and the (provider, prompt) disk cache by monkeypatching each
backend's `call`.
"""

from __future__ import annotations

import pytest

from hems.llm.client import LLMClient, LLMError
from hems.llm.schemas import HVACWTPResponse


def test_unknown_provider_raises():
    with pytest.raises(LLMError):
        LLMClient(provider="openai")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        LLMClient(provider="gemini")


def test_cache_keyed_on_provider_and_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("MAINCODE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cache_path = tmp_path / "cache.json"

    call_count = {"matilda": 0, "gemini": 0}

    def fake_matilda_call(self, prompt, schema):
        call_count["matilda"] += 1
        return {
            "wtp": 1.0,
            "reason": "matilda",
            "echo_indoor_temp_c": 27.0,
            "echo_band_lo_c": 22.0,
            "echo_band_hi_c": 26.0,
        }

    def fake_gemini_call(self, prompt, schema):
        call_count["gemini"] += 1
        return {
            "wtp": 2.0,
            "reason": "gemini",
            "echo_indoor_temp_c": 27.0,
            "echo_band_lo_c": 22.0,
            "echo_band_hi_c": 26.0,
        }

    from hems.llm import client as client_module

    monkeypatch.setattr(client_module._MatildaBackend, "call", fake_matilda_call)
    monkeypatch.setattr(client_module._GeminiBackend, "call", fake_gemini_call)

    matilda_client = LLMClient(provider="matilda", cache_path=cache_path)
    gemini_client = LLMClient(provider="gemini", cache_path=cache_path)

    prompt = "identical prompt text"

    result_1, cached_1 = matilda_client.complete_json(prompt, HVACWTPResponse)
    result_2, cached_2 = matilda_client.complete_json(prompt, HVACWTPResponse)
    result_3, cached_3 = gemini_client.complete_json(prompt, HVACWTPResponse)

    assert not cached_1
    assert cached_2  # second matilda call with the same prompt hits the cache
    assert not cached_3  # same prompt, different provider -- must not cache-collide
    assert result_1.reason == "matilda"
    assert result_3.reason == "gemini"
    assert call_count == {"matilda": 1, "gemini": 1}


def test_gemini_call_passes_temperature_and_seed(monkeypatch):
    """BUILD_SPEC.md §2.3: T=0 alone was not enough for determinism in the
    v3 probe -- the seed must also reach `GenerateContentConfig`. Captures
    the config `_GeminiBackend.call` builds without hitting the network."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured["temperature"] = config.temperature
            captured["seed"] = config.seed

            class FakeResponse:
                text = '{"wtp": 1.0, "reason": "ok", "echo_indoor_temp_c": 27.0, ' \
                       '"echo_band_lo_c": 22.0, "echo_band_hi_c": 26.0}'
                candidates = []

            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    import google.genai as real_genai

    monkeypatch.setattr(real_genai, "Client", FakeClient)

    default_client = LLMClient(provider="gemini")
    default_client.complete_json("prompt one", HVACWTPResponse)
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42

    custom_client = LLMClient(provider="gemini", temperature=0.7, seed=7)
    custom_client.complete_json("prompt two", HVACWTPResponse)
    assert captured["temperature"] == 0.7
    assert captured["seed"] == 7


def test_repeated_identical_prompt_returns_identical_wtp(monkeypatch):
    """End-to-end determinism check (VERIFY & REPORT): two identical calls
    for the same (provider, prompt) return byte-identical wtp -- via the
    disk cache in the general case, and the backend itself is seeded so a
    live cache-miss rerun would also agree (see the temperature/seed test
    above)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    call_count = {"n": 0}

    def fake_call(self, prompt, schema):
        call_count["n"] += 1
        return {
            "wtp": 3.25,
            "reason": "deterministic",
            "echo_indoor_temp_c": 27.0,
            "echo_band_lo_c": 22.0,
            "echo_band_hi_c": 26.0,
        }

    from hems.llm import client as client_module

    monkeypatch.setattr(client_module._GeminiBackend, "call", fake_call)

    client = LLMClient(provider="gemini")
    result_1, _ = client.complete_json("identical prompt", HVACWTPResponse)
    result_2, _ = client.complete_json("identical prompt", HVACWTPResponse)

    assert result_1.wtp == result_2.wtp == 3.25
    assert call_count["n"] == 1  # second call served from cache, not re-sent


def test_model_id_and_sdk_version_recorded(monkeypatch):
    monkeypatch.setenv("MAINCODE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    matilda_client = LLMClient(provider="matilda")
    gemini_client = LLMClient(provider="gemini", model="gemini-3.5-flash-lite")

    assert matilda_client.provider == "matilda"
    assert "matilda-client" in matilda_client.sdk_version

    assert gemini_client.provider == "gemini"
    assert gemini_client.model_id == "gemini-3.5-flash-lite"


def _fake_hvac_payload(wtp=1.0):
    return {"wtp": wtp, "reason": "x", "echo_indoor_temp_c": 27.0,
            "echo_band_lo_c": 22.0, "echo_band_hi_c": 26.0}


def test_seed_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """Phase 3 multi-seed: seed 43 must not be served seed 42's cached answer."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from hems.llm import client as client_module

    monkeypatch.setattr(client_module._GeminiBackend, "call",
                        lambda self, p, s: _fake_hvac_payload(float(self._seed)))
    path = tmp_path / "cache.jsonl"
    a = LLMClient(provider="gemini", seed=42, cache_path=path)
    b = LLMClient(provider="gemini", seed=43, cache_path=path)
    ra, ca = a.complete_json("same prompt", HVACWTPResponse)
    rb, cb = b.complete_json("same prompt", HVACWTPResponse)
    assert (ra.wtp, rb.wtp) == (42.0, 43.0)
    assert not ca and not cb


def test_jsonl_cache_second_run_makes_zero_live_calls(tmp_path, monkeypatch):
    """Re-running identical prompts in a fresh process-equivalent hits disk only."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from hems.llm import client as client_module

    monkeypatch.setattr(client_module._GeminiBackend, "call", lambda self, p, s: _fake_hvac_payload())
    path = tmp_path / "cache.jsonl"
    first = LLMClient(provider="gemini", cache_path=path)
    for i in range(3):
        first.complete_json(f"prompt {i}", HVACWTPResponse)
    assert first.stats == {"live_calls": 3, "cache_hits": 0}
    assert len(path.read_text().splitlines()) == 3  # append-only, one line per call

    client_module._DiskCache._instances.clear()  # simulate a new process
    monkeypatch.setattr(client_module._GeminiBackend, "call",
                        lambda self, p, s: pytest.fail("live call on a cached prompt"))
    second = LLMClient(provider="gemini", cache_path=path)
    for i in range(3):
        second.complete_json(f"prompt {i}", HVACWTPResponse)
    assert second.stats == {"live_calls": 0, "cache_hits": 3}


def test_cache_key_includes_schema_hash_and_max_output_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from hems.llm.schemas import WasherWTPResponse

    path = tmp_path / "cache.jsonl"
    a = LLMClient(provider="gemini", max_output_tokens=2048, cache_path=path)
    b = LLMClient(provider="gemini", max_output_tokens=1024, cache_path=path)
    assert a.cache_key("p", HVACWTPResponse) != b.cache_key("p", HVACWTPResponse)
    assert a.cache_key("p", HVACWTPResponse) != a.cache_key("p", WasherWTPResponse)
    assert a.cache_key("p", HVACWTPResponse) == a.cache_key("p", HVACWTPResponse)


def test_cache_entry_stores_model_version_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import json

    from hems.llm import client as client_module

    def fake_call(self, prompt, schema):
        self.last_model_version = "gemini-3.5-flash-lite-001"
        return _fake_hvac_payload()

    monkeypatch.setattr(client_module._GeminiBackend, "call", fake_call)
    path = tmp_path / "cache.jsonl"
    c = LLMClient(provider="gemini", cache_path=path)
    c.complete_json("p", HVACWTPResponse)
    assert c.last_call["model_version"] == "gemini-3.5-flash-lite-001"
    assert c.last_call["cached"] is False
    entry = json.loads(path.read_text().splitlines()[0])
    assert entry["model_version"] == "gemini-3.5-flash-lite-001"
    assert entry["timestamp"].endswith("+00:00")

    client_module._DiskCache._instances.clear()  # new process: metadata survives the reload
    c2 = LLMClient(provider="gemini", cache_path=path)
    _, cached = c2.complete_json("p", HVACWTPResponse)
    assert cached and c2.last_call["model_version"] == "gemini-3.5-flash-lite-001"
    assert c2.model_versions_seen == {"gemini-3.5-flash-lite-001": 1}


def test_build_llm_client_reads_retries_backoff_and_provider_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from hems.llm.client import build_llm_client

    cfg = {
        "temperature": 0.0,
        "seed": 42,
        "max_retries": 5,
        "backoff": {"base_s": 0.5, "max_s": 4.0, "rate_limited_min_s": 30.0},
        "gemini": {"model": "m-x", "rate_limit_rpm": 9, "max_output_tokens": 777},
    }
    c = build_llm_client("gemini", cfg, None, seed=43)
    assert c.settings["max_retries"] == 5
    assert c.settings["backoff"] == cfg["backoff"]
    assert c.settings["rate_limit_rpm"] == 9
    assert c.settings["model"] == "m-x"
    assert c.settings["max_output_tokens"] == 777
    assert c.settings["seed"] == 43


def test_retry_backoff_follows_config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from hems.llm import client as client_module

    sleeps = []
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    def failing(self, p, s):
        raise RuntimeError("boom")

    monkeypatch.setattr(client_module._GeminiBackend, "call", failing)
    c = LLMClient(provider="gemini", max_retries=3, backoff={"base_s": 1.0, "max_s": 3.0})
    with pytest.raises(LLMError):
        c.complete_json("p", HVACWTPResponse)
    assert sleeps == [1.0, 2.0, 3.0]
