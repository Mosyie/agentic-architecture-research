import dotenv

from langchain.agents        import create_agent as _create_agent
from langchain_core.messages import HumanMessage, AIMessage

from src.core.metrics    import llm_accounting
from src.core.llm_client import get_llm_client


dotenv.load_dotenv()

SYSTEM_PROMPT = (
    "You are a question-answering agent for multi-hop reasoning tasks.\n"
    "Use the available tools to inspect context for the current question.\n"
    "Answer strictly using information from the tool outputs.\n"
    "Use tools before answering.\n"
    "For multi-hop questions, read multiple relevant chunks if needed.\n"
    "The final answer must be short and contain no explanation."
)

class SingleAgent:
    """LangGraph-backed agent"""

    def __init__(
        self,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.max_steps = max_steps
        self.verbose = verbose

        self.llm = get_llm_client()

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def invoke(self, question: str, tools) -> dict:

        agent = _create_agent(self.llm, tools, system_prompt=SYSTEM_PROMPT)

        # recursion_limit caps total graph steps. Each agent "step" in the
        # original = one LLM call + its tool calls, which in LangGraph is two
        # nodes (agent + tools). We give 2x headroom plus a margin so the
        # limit isn't hit before max_steps real reasoning turns.
        config = {"recursion_limit": self.max_steps * 2 + 1}

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=f"Question: {question}")]},
                config=config,
            )
        except Exception as exc:  # graph errors, recursion limit, endpoint errors
            return {
                "answer": "",
                "total_tokens": 0,
                "num_api_calls": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

        messages = result["messages"]

        if self.verbose:
            for m in messages:
                self._log(repr(m))

        total_tokens, num_api_calls = llm_accounting(messages)

        # The final AIMessage with no tool calls holds the answer.
        final_answer = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                final_answer = (m.content or "").strip()
                break

        out = {
            "answer": final_answer,
            "total_tokens": total_tokens,
            "num_api_calls": num_api_calls,
        }
        if not final_answer:
            out["error"] = "No final answer produced"
        return out
