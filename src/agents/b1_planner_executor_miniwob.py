import json

from typing import Literal, NamedTuple, Optional, TypedDict

from pydantic import BaseModel, Field

from langgraph.graph         import END, START, StateGraph
from langchain.agents        import create_agent as _create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.metrics    import llm_accounting
from src.core.llm_client import get_llm_client


EXECUTOR_SYSTEM_PROMPT = (
    "You are an Executor agent controlling a web page.\n"
    "You are given ONE narrow UI sub-task and the current list of interactable "
    "elements (each with a ref id, HTML tag, an optional #id, text, -- for "
    "inputs -- a value, and its position as (left,top widthxheight): the top-left "
    "corner and the width x height size, in page pixels).\n"
    "Use the tools to act: click_element(ref) to click, type_text(text, ref) to "
    "enter text, press_key(key) to press a key on the focused element\n"
    "drag(sx, sy, tx, ty) to drag from page point (sx,sy) to (tx,ty) -- press at "
    "(sx,sy), release at (tx,ty), in page pixels matching the shown positions; to "
    "grab or drop on an element, aim at its center, (left + width/2, top + height/2). "
    "Refer to elements by the ref ids shown (drag takes coordinates, not refs).\n"
    "Accomplish ONLY the given sub-task -- do not improvise beyond it or try to "
    "finish the whole task.\n"
    "When done, reply with a SHORT status describing what you did and the "
    "resulting state (e.g. 'Clicked the date field; a calendar opened on "
    "December 2015')."
)


PLANNER_SYSTEM_PROMPT = (
    "You complete a web UI task by delegating.\n"
    "You have NO tools and you cannot act on the page yourself. You see the goal "
    "and the initial page once; afterwards you only see the Executor's short "
    "status reports.\n"
    "\n"
    "Strategy:\n"
    "- Break the goal into single UI sub-steps.\n"
    "- Issue ONE sub-step at a time. Wait for the Executor's report before "
    "deciding the next one (later steps depend on earlier UI changes).\n"
    "- Build your understanding of the page from the Executor's reports only.\n"
    "\n"
    "Output rules:\n"
    "- To delegate the next action: status='continue' and put the action in "
    "'substep'. The page is NOT yet solved, so your FIRST reply is almost always "
    "'continue'.\n"
    "- To stop: status='done' and leave 'substep' EMPTY. Use this ONLY after the "
    "goal is fully achieved.\n"
    "- Never set status='done' while 'substep' still describes an action to take."
)


class PlannerDecision(BaseModel):
    """Planner's next move: either delegate one sub-step, or finish."""

    status: Literal["continue", "done"] = Field(
        description=(
            "'continue' to issue another UI sub-step, "
            "'done' when the goal is achieved."
        )
    )
    substep: str = Field(
        default="",
        description=(
            "When status is 'continue': one narrow UI sub-task for the "
            "Executor. Empty string when finishing."
        ),
    )


class SubstepLogEntry(NamedTuple):
    substep: str
    report: str


class PlannerExecutorState(TypedDict, total=False):
    instruction: str
    current_observation: str
    substep_log: list[SubstepLogEntry]
    next_substep: str
    done: bool
    terminated: bool
    total_tokens: int
    num_api_calls: int
    num_substeps: int
    planner_steps: int
    error: str


class PlannerExecutorAgent:
    """Two-node planner/executor graph for MiniWoB++ interactive tasks."""

    def __init__(
        self,
        max_planner_steps: int = 8,
        executor_max_steps: int = 10,
    ):
        self.max_planner_steps = max_planner_steps
        self.executor_max_steps = executor_max_steps
        self.planner_llm  = get_llm_client()
        self.executor_llm = get_llm_client()

    def _planner_node(self, state: PlannerExecutorState) -> dict:
        """Ask the planner for either the next sub-step or to finish.

        On any failure (LLM error, unparseable reply) this returns an `error`
        and no `next_substep`/`done`. The router keys off those.
        """
        msgs = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Task (initial page):\n{state['instruction']}"
            ),
        ]

        for entry in state.get("substep_log", []):
            msgs.append(
                AIMessage(
                    content=json.dumps(
                        {"status": "continue", "substep": entry.substep}
                    )
                )
            )
            msgs.append(
                HumanMessage(
                    content=(
                        f"Executor report for '{entry.substep}': {entry.report}\n"
                        "Decide the next sub-step or finish."
                    )
                )
            )

        typed = self.planner_llm.with_structured_output(
            PlannerDecision, method="json_schema", include_raw=True
        )

        try:
            result = typed.invoke(msgs)
        except Exception as exc:
            return {
                "error": f"Planner call failed: {type(exc).__name__}: {exc}",
            }

        ai = result["raw"]
        decision: Optional[PlannerDecision] = result["parsed"]
        print(f"\nPLANNER: {decision!r}")

        planner_tokens, planner_calls = llm_accounting([ai])

        updates = {
            "total_tokens": state.get("total_tokens", 0) + planner_tokens,
            "num_api_calls": state.get("num_api_calls", 0) + planner_calls,
            "planner_steps": state.get("planner_steps", 0) + 1,
        }

        if decision is None:
            updates["error"] = "Planner returned invalid structured output."
            return updates

        substep = (decision.substep or "").strip()

        if decision.status == "done":
            updates["done"] = True
        else:
            updates["next_substep"] = substep

        return updates

    def _executor_node(self, state: PlannerExecutorState, executor_agent) -> dict:
        """Run the tool-using executor on the current sub-step against the env."""
        substep = state.get("next_substep", "").strip()
        current_observation = state.get("current_observation", "")

        if not substep:
            return {
                "substep_log": state.get("substep_log", [])
                + [SubstepLogEntry("(empty sub-step)", "skipped")],
                "num_substeps": state.get("num_substeps", 0) + 1,
                "error": "Planner produced an empty sub-step.",
            }

        print(f"--> DELEGATING SUB-STEP: {substep}")

        report = "(no report)"
        new_observation = current_observation
        terminated = False
        exec_tokens = 0
        exec_calls = 0

        first_message = HumanMessage(
            content=(
                f"Current page:\n{current_observation}\n\nSub-task: {substep}"
            )
        )

        try:
            result = executor_agent.invoke(
                {"messages": [first_message]},
                config={
                    "recursion_limit": self.executor_max_steps * 2 + 1,
                    "max_concurrency": 1,
                },
            )

            messages = result.get("messages", [])
            exec_tokens, exec_calls = llm_accounting(messages)

            for message in reversed(messages):
                if isinstance(message, AIMessage) and not getattr(
                    message, "tool_calls", None
                ):
                    report = (message.content or "").strip() or "(no report)"
                    break

            # The most recent ToolMessage carries the post-action observation
            # (or the "Episode finished" marker). Keep the prior observation if
            # the executor took no action.
            for message in reversed(messages):
                if isinstance(message, ToolMessage):
                    content = message.content or ""
                    new_observation = content if isinstance(content, str) else str(content)
                    if "Episode finished" in new_observation:
                        terminated = True
                    break

        except Exception as exc:
            return {
                "substep_log": state.get("substep_log", [])
                + [SubstepLogEntry(substep, "(executor error)")],
                "current_observation": new_observation,
                "total_tokens": state.get("total_tokens", 0) + exec_tokens,
                "num_api_calls": state.get("num_api_calls", 0) + exec_calls,
                "num_substeps": state.get("num_substeps", 0) + 1,
                "error": f"Executor failed: {type(exc).__name__}: {exc}",
            }

        print(f"<-- EXECUTOR REPORT: {report}")

        return {
            "substep_log": state.get("substep_log", [])
            + [SubstepLogEntry(substep, report)],
            "current_observation": new_observation,
            "terminated": terminated,
            "total_tokens": state.get("total_tokens", 0) + exec_tokens,
            "num_api_calls": state.get("num_api_calls", 0) + exec_calls,
            "num_substeps": state.get("num_substeps", 0) + 1,
        }

    def _route_after_planner(self, state: PlannerExecutorState) -> str:
        """Route to executor unless we errored, finished, or hit the step guard."""
        if state.get("error"):
            return "end"

        if state.get("done"):
            return "end"

        if state.get("planner_steps", 0) >= self.max_planner_steps:
            return "end"

        return "executor"

    def _route_after_executor(self, state: PlannerExecutorState) -> str:
        """Stop as soon as the env episode ends; otherwise re-plan."""
        if state.get("error"):
            return "end"

        if state.get("terminated"):
            return "end"

        return "planner"

    def _build_graph(self, executor_agent):
        graph = StateGraph(PlannerExecutorState)

        graph.add_node("planner", self._planner_node)

        def executor_node(state: PlannerExecutorState) -> dict:
            return self._executor_node(state, executor_agent)

        graph.add_node("executor", executor_node)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "executor": "executor",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {
                "planner": "planner",
                "end": END,
            },
        )

        return graph.compile()

    def invoke(self, instruction: str, tools) -> dict:
        """Run the planner/executor graph for one MiniWoB++ task.

        `instruction` is the formatted initial observation (utterance + DOM).
        Reward / num_actions are read from the shared episode_state by the
        runner; this returns only LLM accounting and the sub-step count.
        """
        executor_agent = _create_agent(
            self.executor_llm,
            tools,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
        )

        graph = self._build_graph(executor_agent)

        initial_state: PlannerExecutorState = {
            "instruction": instruction,
            "current_observation": instruction,
            "substep_log": [],
            "total_tokens": 0,
            "num_api_calls": 0,
            "num_substeps": 0,
            "planner_steps": 0,
        }

        # Each planner step is up to two graph nodes (planner -> executor), so
        # the graph-level recursion budget is ~2x the planner-step budget plus
        # slack for the final planner pass and routing.
        config = {"recursion_limit": self.max_planner_steps * 2 + 5}

        try:
            final_state = graph.invoke(initial_state, config=config)
        except Exception as exc:
            return {
                "total_tokens": 0,
                "num_api_calls": 0,
                "num_substeps": 0,
                "error": f"Graph failed: {type(exc).__name__}: {exc}",
            }

        error = final_state.get("error", "")

        if not error and not final_state.get("done") and not final_state.get("terminated"):
            if final_state.get("planner_steps", 0) >= self.max_planner_steps:
                error = "Maximum planner steps reached."

        output = {
            "total_tokens": final_state.get("total_tokens", 0),
            "num_api_calls": final_state.get("num_api_calls", 0),
            "num_substeps": final_state.get("num_substeps", 0),
        }

        if error:
            output["error"] = error

        return output
