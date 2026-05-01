import json
from src.core.llm_client import get_llm_client
from src.core.tool_environment import ToolEnvironment


class SingleAgentV2:

    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.5-Air-FP8",
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.client = get_llm_client()
        self.model_name = model_name
        self.max_steps = max_steps
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def invoke(self, question: str, tool_env: ToolEnvironment) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a question-answering agent for multi-hop reasoning tasks.\n"
                    "Use the available tools to inspect context for the current question.\n"
                    "Answer strictly using information from the tool outputs.\n"
                    "Use tools before answering.\n"
                    "For multi-hop questions, read multiple relevant chunks if needed.\n"
                    "The final answer must be short and contain no explanation."
                )
            },
            {
                "role": "user",
                "content": f"Question: {question}"
            }
        ]

        num_api_calls = 0
        total_tokens = 0

        for step in range(self.max_steps):
            self._log("\n" + "=" * 80)
            self._log(f"LLM API CALL #{num_api_calls + 1} | STEP {step + 1}")
            self._log("=" * 80)

            latest_messages = messages[-2:] if len(messages) > 2 else messages
            self._log("\nLATEST MESSAGE(S) SENT TO MODEL:")
            self._log(json.dumps(latest_messages, indent=2, ensure_ascii=False, default=str))

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tool_env.tools_schema(),
                tool_choice="auto",
                stream=False
            )

            num_api_calls += 1

            if response.usage:
                total_tokens += response.usage.total_tokens

            message = response.choices[0].message
            message_dict = message.model_dump()

            self._log("\nMODEL RESPONSE MESSAGE:")
            self._log(json.dumps(message_dict, indent=2, ensure_ascii=False, default=str))

            messages.append(message_dict)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name

                    try:
                        arguments = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}

                    self._log("\nTOOL CALL REQUESTED:")
                    self._log(f"Tool: {tool_name}")
                    self._log(f"Arguments: {json.dumps(arguments, indent=2, ensure_ascii=False)}")

                    tool_result = tool_env.execute_tool(tool_name, arguments)

                    self._log("\nTOOL RESULT:")
                    self._log(json.dumps(tool_result, indent=2, ensure_ascii=False, default=str))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    })

                continue

            final_answer = (message.content or "").strip()

            return {
                "answer": final_answer,
                "total_tokens": total_tokens,
                "num_api_calls": num_api_calls
            }

        return {
            "answer": "",
            "total_tokens": total_tokens,
            "num_api_calls": num_api_calls,
            "error": "Max steps reached without final answer"
        }