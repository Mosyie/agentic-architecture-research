import json

from typing import NamedTuple, Optional, TypedDict

from pydantic import BaseModel, Field

from langgraph.graph         import END, START, StateGraph
from langchain.agents        import create_agent as _create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.metrics    import llm_accounting
from src.core.llm_client import get_llm_client
from src.core.run_logger import format_action_trace


ACTOR_SYSTEM_PROMPT = (
    "You are a web UI agent that completes tasks on a small web page.\n"
    "You are given an instruction and a list of interactable DOM elements, each "
    "with a ref id, HTML tag, an optional #id, text, (for inputs) a value, and "
    "its position as (left,top widthxheight): the top-left corner and the "
    "width x height size, in page pixels.\n"
    "Use the tools to act: click_element(ref) to click, type_text(text, ref) to enter text, "
    "press_key(key) to press a key (or key combinations) on the focused element, "
    "drag(sx, sy, tx, ty) to drag from page point (sx,sy) to (tx,ty).\n"
    "drag coordinates are page pixels in the same system as the shown positions; to "
    "grab or drop on an element, aim at its center, (left + width/2, top + height/2).\n"
    "After each action you receive an updated element list. Inspect it and continue.\n"
    "Refer to elements only by the ref ids shown in the latest observation.\n"
    "Take the minimum actions needed to satisfy the instruction, then stop.\n"
    "When you believe the task is complete or the episode reports it is finished, "
    "stop calling tools and reply with a SHORT report of what you did and the "
    "final page state (e.g. 'Typed \"abc\" into the text field and clicked Submit')."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a strict Critic agent reviewing a web UI agent (the Actor).\n"
    "The Actor claims the task is done, but the episode has NOT terminated. "
    "The environment ends the episode the moment it judges the task (success "
    "or failure), so the task is genuinely NOT done -- if the Actor were "
    "right, you would not be consulted. The page is still actionable.\n"
    "\n"
    "You are given the task instruction (with the initial page), the Actor's "
    "action trace, the Actor's report, and the CURRENT page state.\n"
    "\n"
    "Diagnose, against the current page, why the task is not done:\n"
    "1. Was the goal misread or only partially achieved?\n"
    "2. Is a final step missing?\n"
    "3. Did any action have no effect?\n"
    "\n"
    "Give precise, actionable feedback: what is wrong and what concrete UI "
    "step(s) remain. Justify the critique from the evidence."
)


class ToolCallEntry(NamedTuple):
    tool_input: str
    tool_output: str


class ActorCriticState(TypedDict, total=False):
    instruction: str
    current_observation: str

    action_trace: list[ToolCallEntry]

    actor_report: str
    critic_feedback: str

    terminated: bool

    total_tokens: int
    num_api_calls: int

    actor_steps: int
    critic_steps: int

    error: str


class CriticFeedback(BaseModel):
    """Critic's diagnosis of why the Actor's 'done' claim is premature."""

    feedback: str = Field(
        description=(
            "What is wrong and which concrete UI step(s) remain, justified "
            "from the evidence."
        )
    )


def _format_trace(action_trace: list[ToolCallEntry]) -> str:
    if not action_trace:
        return "(no actions taken)"
    return "\n".join(f"- {e.tool_input} -> {e.tool_output}" for e in action_trace)


class ActorCriticAgent:
    """Actor + Critic loop for MiniWoB++ interactive tasks.

    The Actor acts freely on the page (like the A1 single agent). When it
    stops and reports the task as done, the loop checks the environment:

    - episode terminated -> nothing left to revise, the critic is skipped
      (a critique after the terminal action cannot change the reward);
    - episode still running -> the env has not judged the task yet, so the
      done-claim is premature by definition. The Critic diagnoses what is
      wrong or missing and always sends feedback for another acting round.

    After ``max_rounds`` critic rejections the state stands as-is.
    """

    def __init__(self, max_rounds: int = 3, actor_max_steps: int = 15):
        self.actor_llm  = get_llm_client()
        self.critic_llm = get_llm_client()
        self.max_rounds = max_rounds
        self.actor_max_steps = actor_max_steps

    def _actor_node(self, state: ActorCriticState, actor_agent) -> dict:
        messages = [HumanMessage(content=state["instruction"])]

        if state.get("critic_feedback"):
            messages.append(
                HumanMessage(
                    content=(
                        "You reported the task as done, but the episode has "
                        "not finished.\n"
                        f"Critic feedback: {state['critic_feedback']}\n"
                        "Continue working on the page and fix this.\n\n"
                        f"Current page:\n{state.get('current_observation', '')}"
                    )
                )
            )

        new_observation = state.get("current_observation", state["instruction"])
        terminated = False

        try:
            result = actor_agent.invoke(
                {"messages": messages},
                config={
                    "recursion_limit": self.actor_max_steps * 2 + 1,
                    "max_concurrency": 1,
                },
            )
            messages_out = result.get("messages", [])

            tokens, calls = llm_accounting(messages_out)

            report = ""
            for m in reversed(messages_out):
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                    report = (m.content or "").strip()
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

            for m in reversed(messages_out):
                if isinstance(m, ToolMessage):
                    content = m.content or ""
                    new_observation = content if isinstance(content, str) else str(content)
                    if (
                        "Episode finished" in new_observation
                        or "Episode already finished" in new_observation
                    ):
                        terminated = True
                    break

        except Exception as exc:  # graph errors, recursion limit, endpoint errors
            return {
                "error": f"Actor failed: {type(exc).__name__}: {exc}"
            }

        print(format_action_trace(messages_out))
        print(f"[ACTOR] {report}")

        return {
            "actor_report": report,
            "action_trace": state.get("action_trace", []) + new_entries,
            "current_observation": new_observation,
            "terminated": terminated,
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "num_api_calls": state.get("num_api_calls", 0) + calls,
            "actor_steps": state.get("actor_steps", 0) + 1,
            "critic_feedback": "",
        }

    def _critic_node(self, state: ActorCriticState) -> dict:
        trace = _format_trace(state.get("action_trace", []) or [])

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Task (initial page):\n{state['instruction']}\n\n"
                    f"Action Trace:\n{trace}\n\n"
                    f"Actor Report:\n{state.get('actor_report', '')}\n\n"
                    f"Current page:\n{state.get('current_observation', '')}\n"
                )
            ),
        ]

        typed = self.critic_llm.with_structured_output(
            CriticFeedback, method="json_schema", include_raw=True
        )

        try:
            result = typed.invoke(messages)
        except Exception as exc:
            return {
                "error": f"Critic failed: {type(exc).__name__}: {exc}"
            }

        ai = result["raw"]
        critique: Optional[CriticFeedback] = result["parsed"]
        print(f"[CRITIC] {critique!r}")

        tokens, calls = llm_accounting([ai])

        updates = {
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "num_api_calls": state.get("num_api_calls", 0) + calls,
            "critic_steps": state.get("critic_steps", 0) + 1,
        }

        if critique is None:
            updates["error"] = "Critic returned invalid structured output"
            return updates

        updates["critic_feedback"] = (critique.feedback or "").strip() or (
            "The episode has not finished, so the task is not complete. "
            "Re-examine the page and finish the remaining steps."
        )

        return updates

    def _route_after_actor(self, state: ActorCriticState) -> str:
        if state.get("error"):
            return "end"

        if state.get("terminated"):
            print("[CRITIC] skipped: episode already terminated")
            return "end"

        return "critic"

    def _route_after_critic(self, state: ActorCriticState) -> str:
        if state.get("error"):
            return "end"

        # Rejection cap reached: the current state stands as-is.
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
        graph.add_conditional_edges(
            "actor",
            self._route_after_actor,
            {
                "critic": "critic",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "actor": "actor",
                "end": END,
            },
        )

        return graph.compile()

    def invoke(self, instruction: str, tools) -> dict:
        """Run the actor/critic graph for one MiniWoB++ task.

        `instruction` is the formatted initial observation (utterance + DOM).
        Reward / num_actions are read from the shared episode_state by the
        runner; this returns only LLM accounting and the round count.
        """
        actor_agent = _create_agent(
            self.actor_llm,
            tools,
            system_prompt=ACTOR_SYSTEM_PROMPT,
        )

        graph = self._build_graph(actor_agent)

        initial_state: ActorCriticState = {
            "instruction": instruction,
            "current_observation": instruction,
            "action_trace": [],
            "total_tokens": 0,
            "num_api_calls": 0,
            "actor_steps": 0,
            "critic_steps": 0,
        }

        # Each round is up to two graph nodes (actor -> critic), so the
        # graph-level recursion budget is ~2x the round budget plus slack.
        config = {"recursion_limit": self.max_rounds * 2 + 5}

        try:
            final_state = graph.invoke(initial_state, config=config)
        except Exception as exc:
            return {
                "total_tokens": 0,
                "num_api_calls": 0,
                "num_rounds": 0,
                "error": f"Graph failed: {type(exc).__name__}: {exc}",
            }

        output = {
            "total_tokens": final_state.get("total_tokens", 0),
            "num_api_calls": final_state.get("num_api_calls", 0),
            "num_rounds": final_state.get("actor_steps", 0),
        }

        error = final_state.get("error", "")
        if error:
            output["error"] = error

        return output
