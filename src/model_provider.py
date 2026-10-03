from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


# Aliases mapping for common typos and variations
_PROVIDER_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "anthropic2": "anthropic",
    "openai2": "openai",
    "gemini2": "gemini",
    "ollama2": "ollama",
    "openrouter2": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`.

    Returns the normalized provider name or the original value if no alias is found.
    """
    value_lower = value.lower().strip()
    return _PROVIDER_ALIASES.get(value_lower, value_lower)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Providers:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`

    Returns None if the required API key is not available for the provider.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        if not config.base_url:
            raise ValueError("Custom provider requires CUSTOM_BASE_URL environment variable")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not config.api_key:
            raise ValueError("Gemini provider requires GEMINI_API_KEY environment variable")
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not config.api_key:
            raise ValueError("Anthropic provider requires ANTHROPIC_API_KEY environment variable")
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            anthropic_api_key=config.api_key,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = config.base_url or "http://localhost:11434"
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
        )

    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        if not config.api_key:
            raise ValueError("OpenRouter provider requires OPENROUTER_API_KEY environment variable")
        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            openrouter_api_key=config.api_key,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")
