from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# Set the API key and the base URL (with the /v1 endpoint)
client = OpenAI(
    base_url="http://mobydick.elte-dh.hu:23432/v1",
    api_key=dotenv.get_key(dotenv.find_dotenv(),"API_KEY")
)

chat_completion = client.chat.completions.create(
    model="zai-org/GLM-4.5-Air-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is the purpose of life"}
    ],
    stream=False
)

print(chat_completion)