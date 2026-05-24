from src.core.llm_client import get_llm_client

class BaseLineAgent:
    """
    Baseline
    Single LLM working on the input task.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3.6-27B"):
        self.client = get_llm_client()
        self.model_name = model_name
        
    def invoke(self, prompt: str) -> dict:
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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        # Manually returning the data needed for your cost measurements
        return {
            "answer": response.choices[0].message.content,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "api_calls": 1
        }