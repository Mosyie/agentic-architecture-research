import json
import traceback
from typing import Literal, NamedTuple, Optional, TypedDict

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END

from langchain.agents        import create_agent as _create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from src.core.llm_client import get_llm_client
from src.core.metrics    import llm_accounting


ACTOR_SYSTEM_PROMPT = (
    "You are the Actor agent.\n"
    "You answer questions using tools when needed.\n"
    "You MUST use tools before concluding if uncertainty exists.\n"
    "Return a final concise answer only.\n"
    "No explanations unless required by the question."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a strict Critic agent.\n"
    "Your job is to verify whether the Actor's answer is fully supported by:\n"
    "- the original question\n"
    "- the tool outputs / retrieved context\n"
    "\n"
    "You must check:\n"
    "1. Is every factual claim supported by tool/context evidence?\n"
    "2. Are there missing reasoning hops?\n"
    "3. Did the Actor hallucinate or infer unsupported facts?\n"
    "\n"
    "Accept only if the answer is fully supported. Otherwise reject and\n"
    "provide precise feedback on the missing information or correction."
)


class ToolCallEntry(NamedTuple):
    tool_input: str
    tool_output: str


class ActorCriticState(TypedDict, total=False):
    question: str

    tool_trace: list[ToolCallEntry]

    actor_answer: str
    critic_feedback: str

    total_tokens: int
    num_api_calls: int

    actor_steps: int
    critic_steps: int

    final_answer: str
    error: str


class CriticVerdict(BaseModel):
    """Critic's verdict on whether the Actor's answer is fully supported."""

    status: Literal["accept", "reject"] = Field(
        description=(
            "'accept' if the answer is fully supported by the tool evidence; "
            "'reject' otherwise."
        )
    )
    feedback: str = Field(
        default="",
        description=(
            "When status is 'reject': precise missing info or correction. "
            "Empty string when accepted."
        ),
    )


def _format_trace(tool_trace: list[ToolCallEntry]) -> str:
    if not tool_trace:
        return "(no tool calls)"
    return "\n".join(f"- {e.tool_input} -> {e.tool_output}" for e in tool_trace)


class ActorCriticAgent:
    """Actor + Critic loop. The Actor answers using tools; the Critic
    verifies the answer against the tool trace and either accepts or
    sends feedback for a retry.

    Subclasses can override ``actor_system_prompt`` and ``critic_system_prompt``
    to inject additional guidance without re-implementing the graph.
    """

    def __init__(self, max_rounds: int = 5, verbose: bool = False):
        self.actor_llm  = get_llm_client()
        self.critic_llm = get_llm_client()
        self.max_rounds = max_rounds
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @property
    def actor_system_prompt(self) -> str:
        return ACTOR_SYSTEM_PROMPT

    @property
    def critic_system_prompt(self) -> str:
        return CRITIC_SYSTEM_PROMPT

    def _actor_node(self, state: ActorCriticState, actor_agent) -> dict:
        messages = [HumanMessage(content=state["question"])]

        if state.get("critic_feedback"):
            messages.append(
                HumanMessage(
                    content=(
                        "Previous answer was rejected.\n"
                        f"Critic feedback: {state['critic_feedback']}\n"
                        "Fix all issues before answering again."
                    )
                )
            )

        try:
            result = actor_agent.invoke({"messages": messages})
            messages_out = result.get("messages", [])

            tokens, calls = llm_accounting(messages_out)

            answer = ""
            for m in reversed(messages_out):
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                    answer = (m.content or "").strip()
                    break

            new_entries: list[ToolCallEntry] = []
            tool_calls_by_id: dict[str, dict] = {}
            for m in messages_out:
                if isinstance(m, AIMessage):
                    for tc in (getattr(m, "tool_calls", None) or []):
                        tool_calls_by_id[tc["id"]] = tc
                elif isinstance(m, ToolMessage):
                    tc = tool_calls_by_id.get(m.tool_call_id)
                    if tc is None:
                        continue
                    args_str = json.dumps(tc.get("args", {}), default=str, ensure_ascii=False)
                    tool_input = f"{tc.get('name', '<tool>')}({args_str})"
                    output = m.content
                    tool_output = output if isinstance(output, str) else str(output)
                    new_entries.append(ToolCallEntry(tool_input=tool_input, tool_output=tool_output))

        except Exception as exc:
            return {
                "error": f"Actor failed: {type(exc).__name__}: {exc}"
            }

        self._log(f"[ACTOR] {answer}")

        return {
            "actor_answer": answer,
            "tool_trace": state.get("tool_trace", []) + new_entries,
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "num_api_calls": state.get("num_api_calls", 0) + calls,
            "actor_steps": state.get("actor_steps", 0) + 1,
            "critic_feedback": "",
        }

    def _critic_node(self, state: ActorCriticState) -> dict:
        trace = _format_trace(state.get("tool_trace", []) or [])

        messages = [
            SystemMessage(content=self.critic_system_prompt),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Actor Answer:\n{state.get('actor_answer','')}\n\n"
                    f"Tool Trace:\n{trace}\n"
                )
            ),
        ]

        typed = self.critic_llm.with_structured_output(
            CriticVerdict, include_raw=True
        )

        try:
            result = typed.invoke(messages)
        except Exception as exc:
            traceback.print_exc()
            return {
                "error": (
                    f"Critic failed: "
                    f"{type(exc).__name__}: {repr(exc)}"
                )
            }

        ai = result["raw"]
        verdict: Optional[CriticVerdict] = result["parsed"]
        self._log(f"[CRITIC] {verdict!r}")

        tokens, calls = llm_accounting([ai])

        updates = {
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "num_api_calls": state.get("num_api_calls", 0) + calls,
            "critic_steps": state.get("critic_steps", 0) + 1,
        }

        if verdict is None:
            updates["error"] = "Critic returned invalid structured output"
            return updates

        if verdict.status == "accept":
            updates["final_answer"] = state.get("actor_answer", "")
        else:
            updates["critic_feedback"] = (verdict.feedback or "").strip()

        return updates

    def _route(self, state: ActorCriticState) -> str:
        if state.get("error"):
            return "end"

        if state.get("final_answer"):
            return "end"

        if state.get("critic_steps", 0) >= self.max_rounds:
            return "end"

        return "actor"

    def _build_graph(self, actor_agent):
        graph = StateGraph(ActorCriticState)

        def actor_node(state: ActorCriticState) -> dict:
            return self._actor_node(state, actor_agent)

        graph.add_node("actor", actor_node)
        graph.add_node("critic", self._critic_node)

        graph.add_edge(START, "actor")
        graph.add_edge("actor", "critic")

        graph.add_conditional_edges(
            "critic",
            self._route,
            {
                "actor": "actor",
                "end": END,
            },
        )

        return graph.compile()

    def invoke(self, question: str, tools) -> dict:
        actor_agent = _create_agent(
            self.actor_llm,
            tools,
            system_prompt=self.actor_system_prompt,
        )

        graph = self._build_graph(actor_agent)

        initial_state: ActorCriticState = {
            "question": question,
            "tool_trace": [],
            "total_tokens": 0,
            "num_api_calls": 0,
            "actor_steps": 0,
            "critic_steps": 0,
        }

        try:
            final_state = graph.invoke(
                initial_state,
                config={"recursion_limit": self.max_rounds * 2 + 5},
            )
        except Exception as exc:
            return {
                "answer": "",
                "tool_trace": [],
                "total_tokens": 0,
                "num_api_calls": 0,
                "error": f"Graph failed: {exc}",
            }

        return {
            "answer": final_state.get("final_answer", ""),
            "tool_trace": final_state.get("tool_trace", []),
            "error": final_state.get("error", ""),
            "total_tokens": final_state.get("total_tokens", 0),
            "num_api_calls": final_state.get("num_api_calls", 0),
        }
