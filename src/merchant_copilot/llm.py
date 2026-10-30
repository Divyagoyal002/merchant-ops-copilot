import os

from langchain_core.language_models import BaseChatModel

from merchant_copilot.config import ConfigError

PROVIDER_KEYS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY", "gemini": "GOOGLE_API_KEY"}


def build_llm(model_spec: str, max_tokens: int = 2048) -> BaseChatModel:
    provider, model = model_spec.split(":", 1)
    if provider not in PROVIDER_KEYS:
        raise ConfigError(f"Unknown provider '{provider}', use one of {', '.join(PROVIDER_KEYS)}")
    if not os.getenv(PROVIDER_KEYS[provider]):
        raise ConfigError(f"{model_spec} needs {PROVIDER_KEYS[provider]} in .env")

    # temperature=0 keeps tool choice repeatable, which the evals rely on.
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0, max_tokens=max_tokens, max_retries=4)

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            max_retries=4,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=0, max_output_tokens=max_tokens, max_retries=4)
