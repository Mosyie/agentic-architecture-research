import json
import traceback
from typing import NamedTuple, TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from langchain.agents        import create_agent as _create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    "Return ONLY JSON:\n"
    "{\n"
    '  "status": "accept" | "reject",\n'
    '  "feedback": "<only if reject: precise missing info or correction>"\n'
    "}\n"
    "No markdown, no extra text."
)

class ScratchpadEntry(NamedTuple):
    tool_input: str
    tool_output: str

class ActorCriticState(TypedDict, total=False):
    question: str

    scratchpad: list[ScratchpadEntry]

    actor_answer: str
    critic_feedback: str

    total_tokens: int
    num_api_calls: int

    actor_steps: int
    critic_steps: int

    final_answer: str
    error: str

class CriticDecision(TypedDict, total=False):
    status: str
    feedback: str

def _parse_critic_json(text: str) -> Optional[CriticDecision]:
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "").strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    status = data.get("status")
    if status not in {"accept", "reject"}:
        return None

    return {
        "status": status,
        "feedback": data.get("feedback", "").strip() if data.get("feedback") else ""
    }

class ActorCriticAgent:
    def __init__(self, max_rounds: int = 5, verbose: bool = False):
        self.actor_llm  = get_llm_client()
        self.critic_llm = get_llm_client()
        self.max_rounds = max_rounds
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _actor_node(self, state: ActorCriticState, actor_agent) -> dict:
        messages = [
            SystemMessage(content=ACTOR_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]

        # Inject critic feedback if retrying
        if state.get("critic_feedback"):
            messages.append(
                AIMessage(
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
                if isinstance(m, AIMessage):
                    answer = (m.content or "").strip()
                    break

        except Exception as exc:
            return {
                "error": f"Actor failed: {type(exc).__name__}: {exc}"
            }

        self._log(f"[ACTOR] {answer}")

        return {
            "actor_answer": answer,
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "num_api_calls": state.get("num_api_calls", 0) + calls,
            "actor_steps": state.get("actor_steps", 0) + 1,
            "critic_feedback": ""
        }


    def _critic_node(self, state: ActorCriticState) -> dict:
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Actor Answer:\n{state.get('actor_answer','')}\n\n"
                    f"Tool Trace:\n{state.get('scratchpad','')}\n"
                )
            ),
        ]

        try:
            ai = self.critic_llm.invoke(messages)
        except Exception as exc:
              traceback.print_exc()

              return {
                  "error": (
                      f"Critic failed: "
                      f"{type(exc).__name__}: {repr(exc)}"
                  )
              }

        self._log(f"[CRITIC RAW] {ai.content}")

        decision = _parse_critic_json(ai.content or "")
        if not decision:
            return {"error": "Critic returned invalid JSON"}

        if decision["status"] == "accept":
            return {
                "final_answer": state.get("actor_answer", ""),
                "critic_steps": state.get("critic_steps", 0) + 1
            }

        return {
            "critic_feedback": decision.get("feedback", ""),
            "critic_steps": state.get("critic_steps", 0) + 1
        }

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
            system_prompt=ACTOR_SYSTEM_PROMPT,
        )

        graph = self._build_graph(actor_agent)

        initial_state: ActorCriticState = {
            "question": question,
            "scratchpad": [],
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
                "error": f"Graph failed: {exc}",
            }

        return {
            "answer": final_state.get("final_answer", ""),
            "error": final_state.get("error", ""),
            "total_tokens": final_state.get("total_tokens", 0),
            "num_api_calls": final_state.get("num_api_calls", 0),
        }
