import os
import dotenv

from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

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


def get_llm_client() -> ChatOpenAI:
    """Returns a configured LangChain chat model for the endpoint"""
    params = {
        "model": "Qwen/Qwen3.6-27B",
        "temperature": 0,
        "timeout": 30,
        "max_retries": 5,
        **get_llm_config(),
    }
    return ChatOpenAI(**params)
