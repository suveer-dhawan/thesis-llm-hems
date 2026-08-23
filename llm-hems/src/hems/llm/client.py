"""Provider-agnostic LLM wrapper.

Responsibility (BUILD_SPEC.md §14): wraps the two chosen provider
SDKs behind one interface so neither leaks into agent logic. Reads
API keys from `.env` (gitignored). Should support a cached/memoised
call path (§15 cost guard) and a `--dry-run` stub mode for pipeline
testing without spend. No client logic implemented yet -- scaffolding
only.
"""
