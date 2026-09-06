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
            "echo_is_cpp": False,
        }

    def fake_gemini_call(self, prompt, schema):
        call_count["gemini"] += 1
        return {
            "wtp": 2.0,
            "reason": "gemini",
            "echo_indoor_temp_c": 27.0,
            "echo_band_lo_c": 22.0,
            "echo_band_hi_c": 26.0,
            "echo_is_cpp": False,
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


def test_model_id_and_sdk_version_recorded(monkeypatch):
    monkeypatch.setenv("MAINCODE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    matilda_client = LLMClient(provider="matilda")
    gemini_client = LLMClient(provider="gemini", model="gemini-3.5-flash-lite")

    assert matilda_client.provider == "matilda"
    assert "matilda-client" in matilda_client.sdk_version

    assert gemini_client.provider == "gemini"
    assert gemini_client.model_id == "gemini-3.5-flash-lite"
