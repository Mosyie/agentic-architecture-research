import os
import dotenv

from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


# Callbacks attached to every model returned by get_llm_client(). A runner can
# register a handler (e.g. an LLM I/O logger) here BEFORE building its agents so
# all LLM calls -- planner, executor, single agent -- are captured uniformly.
_RUN_CALLBACKS: list = []


def add_llm_callback(handler) -> None:
    """Register a callback handler for subsequently-created LLM clients."""
    if handler not in _RUN_CALLBACKS:
        _RUN_CALLBACKS.append(handler)


def remove_llm_callback(handler) -> None:
    """Unregister a previously added callback handler."""
    if handler in _RUN_CALLBACKS:
        _RUN_CALLBACKS.remove(handler)


def get_llm_config() -> dict:
    base_url = os.environ.get("API_URL")
    api_key = os.environ.get("API_KEY")
    if not base_url:
        raise RuntimeError(
            "API_URL is not set. Check your .env file — without it, "
            "requests fall back to the public OpenAI API and may hang."
        )
    if not api_key:
        raise RuntimeError("API_KEY is not set. Check your .env file.")
    return {"base_url": base_url, "api_key": api_key}


def get_llm_client():
    """Returns a configured LangChain chat model for the endpoint.

    Set the MOCK_LLM env var to get a network-free scripted model instead —
    useful for smoke-testing the pipeline when the LLM endpoint is down.
    """
    callbacks = list(_RUN_CALLBACKS) or None

    if os.environ.get("MOCK_LLM"):
        from src.core.mock_llm import MockToolCallingChat

        return MockToolCallingChat(callbacks=callbacks)

    params = {
        "model": "Qwen/Qwen3.6-27B",
        "temperature": 0,
        "timeout": 30,
        "max_retries": 5,
        "callbacks": callbacks,
        **get_llm_config(),
    }
    return ChatOpenAI(**params)
