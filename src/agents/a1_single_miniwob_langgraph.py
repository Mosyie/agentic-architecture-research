import dotenv

from langchain.agents        import create_agent as _create_agent
from langchain_core.messages import HumanMessage

from src.core.metrics    import llm_accounting
from src.core.llm_client import get_llm_client


dotenv.load_dotenv()

SYSTEM_PROMPT = (
    "You are a web UI agent that completes tasks on a small web page.\n"
    "You are given an instruction and a numbered list of interactable DOM elements, "
    "each with a ref id, tag, text, and value.\n"
    "Use the tools to act: click_element(ref) to click, type_text(text, ref) to enter text.\n"
    "After each action you receive an updated element list. Inspect it and continue.\n"
    "Refer to elements only by the ref ids shown in the latest observation.\n"
    "Take the minimum actions needed to satisfy the instruction, then stop.\n"
    "When the task is complete or the episode reports it is finished, "
    "stop calling tools and reply 'done'."
)

class SingleAgent:
    """LangGraph-backed agent for MiniWoB++ interactive tasks"""

    def __init__(
        self,
        max_steps: int = 15,
        verbose: bool = False,
    ):
        self.max_steps = max_steps
        self.verbose = verbose

        self.llm = get_llm_client()

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def invoke(self, instruction: str, tools) -> dict:

        agent = _create_agent(self.llm, tools, system_prompt=SYSTEM_PROMPT)

        # recursion_limit caps total graph steps. Each agent "step" is one LLM
        # call + its tool calls, which in LangGraph is two nodes (agent + tools).
        # We give 2x headroom plus a margin so the limit isn't hit before
        # max_steps real interaction turns. The tools also short-circuit once
        # the episode terminates, so this is an upper bound, not a target.
        config = {"recursion_limit": self.max_steps * 2 + 1}

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=instruction)]},
                config=config,
            )
        except Exception as exc:  # graph errors, recursion limit, endpoint errors
            return {
                "total_tokens": 0,
                "num_api_calls": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

        messages = result["messages"]

        if self.verbose:
            for m in messages:
                self._log(repr(m))

        total_tokens, num_api_calls = llm_accounting(messages)

        # The reward / action count are read from the shared episode_state by the
        # runner; the agent only reports LLM accounting here.
        return {
            "total_tokens": total_tokens,
            "num_api_calls": num_api_calls,
        }
