from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Paths for the repo root, dataset directory, and state directory.
    Compact-memory settings such as threshold and number of messages to keep.
    Provider settings for openai, custom, gemini, anthropic, ollama, and openrouter.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env file if it exists
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Create state directory if it doesn't exist
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Determine provider and model from environment variables
    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Normalize provider name
    from model_provider import normalize_provider
    provider = normalize_provider(provider)

    # Build model configuration
    model = _build_model_config(provider, model_name)

    # Judge model configuration (same provider for consistency)
    judge_provider = os.getenv("JUDGE_PROVIDER", provider)
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_model = _build_model_config(normalize_provider(judge_provider), judge_model_name)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=2000,
        compact_keep_messages=4,
        model=model,
        judge_model=judge_model,
    )


def _build_model_config(provider: str, model_name: str) -> ProviderConfig:
    """Build ProviderConfig from environment variables based on provider type."""
    api_key = None
    base_url = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "custom":
        base_url = os.getenv("CUSTOM_BASE_URL")
        api_key = os.getenv("CUSTOM_API_KEY")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=0.7,
        api_key=api_key,
        base_url=base_url,
    )
