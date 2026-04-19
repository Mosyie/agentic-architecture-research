from datasets import load_dataset
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

# select first 5 rows
ds_small = ds.select(range(5))

# keep only required columns
cols = ["level", "question", "answer", "context", "supporting_facts"]
ds_small = ds_small.remove_columns([c for c in ds.column_names if c not in cols])

# convert to pandas
df = ds_small.to_pandas()

# Set the API key and the base URL (with the /v1 endpoint)
client = OpenAI(
    base_url="http://mobydick.elte-dh.hu:23432/v1",
    api_key=dotenv.get_key(dotenv.find_dotenv(),"API_KEY")
)

def invoke(prompt: str):
    system_prompt = """
    You are a question-answering agent for multi-hop reasoning tasks.

    You are given:
    - A question
    - A context consisting of multiple entries formatted as:
    Title - Text;
    Title - Text;
    ...

    Your task:
    - Find the answer strictly using the provided context.
    - Perform multi-step reasoning if needed by combining information across multiple entries.
    - Use only the context — do not rely on external knowledge.

    Guidelines:
    - Identify the relevant titles and extract key facts.
    - Chain information across entries when necessary.
    - Be concise and precise.

    Output:
    - Return ONLY the final answer (a short span or phrase).
    - Do NOT include explanations, reasoning, or extra text.
    """

    chat_completion = client.chat.completions.create(
        model="zai-org/GLM-4.5-Air-FP8",
        messages=[
            {"role": "system", "content": system_prompt },
            {"role": "user", "content": prompt}
        ],
        stream=False
    )

    print(chat_completion)

def format_context(context):
    formatted = []

    for title, sentences in zip(context["title"], context["sentences"]):
        text = " ".join(sentences)
        formatted.append(f"{title} - {text}")

    return ";\n".join(formatted)

def format_prompt(question: str, context: str):
    return question + "\n\nContext:\n" + format_context(context)


invoke(format_prompt(df["question"][4], df["context"][4]))