import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

def get_llm_client() -> OpenAI:
    """Returns a configured OpenAI client."""
    return OpenAI(
        base_url="http://mobydick.elte-dh.hu:23432/v1",
        api_key=os.environ.get("API_KEY")
    )