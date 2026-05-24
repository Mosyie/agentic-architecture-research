import json
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent as _create_agent
from src.core.llm_client import get_llm_client
from src.core.metrics import llm_accounting

EXECUTOR_SYSTEM_PROMPT = (
    "You are an Executor agent. You are given ONE narrow sub-question.\n"
    "Use the available tools to inspect context chunks and find the answer.\n"
    "Answer strictly using information from the tool outputs.\n"
    "Use tools before answering. Read multiple chunks if needed.\n"
    "Return ONLY the short, finalized answer to the sub-question -- no "
    "explanation, no raw chunk text. If the context does not contain the "
    "answer, reply exactly: NOT FOUND."
)

PLANNER_SYSTEM_PROMPT = (
    "You are a Planner agent solving a multi-hop question by delegating.\n"
    "You CANNOT see the source documents and you have NO tools. You work only\n"
    "by issuing one narrow sub-question at a time to an Executor agent, which\n"
    "looks things up and returns a short factual answer.\n"
    "\n"
    "Strategy:\n"
    "- Break the main question into single-hop sub-questions.\n"
    "- Issue ONE sub-question at a time. Wait for the Executor's answer before\n"
    "  deciding the next one (later hops often depend on earlier answers).\n"
    "- Build understanding from the Executor's answers only.\n"
    "- When you have enough information, output the final answer.\n"
    "\n"
    "You MUST reply with a single JSON object and nothing else, in one of two\n"
    "forms:\n"
    '  {"status": "continue", "subquery": "<one narrow sub-question>"}\n'
    '  {"status": "done", "answer": "<short final answer, no explanation>"}\n'
    "Do not wrap the JSON in markdown fences. Do not add text around it."
)


def _parse_planner_json(text: str) -> Optional[dict]:
    """Best-effort parse of the planner's JSON reply (tolerates fences/prose)."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------

class PEState(TypedDict, total=False):
    question: str            # the main HotpotQA question
    scratchpad: list         # list of (subquery, exec_answer) the planner has seen
    decision: dict           # planner's latest parsed decision
    next_subquery: str       # sub-query to hand the executor this round
    final_answer: str
    total_tokens: int
    num_api_calls: int
    num_subqueries: int
    planner_steps: int       # guard against infinite planner looping
    error: str


class PlannerExecutorAgent:
    """Builds and runs the two-node planner/executor graph."""

    def __init__(
        self,
        max_planner_steps: int = 8,
        executor_max_steps: int = 10,
        verbose: bool = False,
    ):
        self.max_planner_steps = max_planner_steps
        self.executor_max_steps = executor_max_steps
        self.verbose = verbose
        self.llm = get_llm_client()
        self._tool_env = None
        self._graph = self._build_graph()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # --- node functions ----------------------------------------------------
    def _planner_node(self, state: PEState) -> dict:
        """Build the planner prompt from scratchpad, get one decision."""
        msgs = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Main question: {state['question']}"),
        ]

        for sub, ans in state.get("scratchpad", []):
            msgs.append(AIMessage(content=json.dumps(
                {"status": "continue", "subquery": sub}
            )))
            msgs.append(HumanMessage(content=(
                f"Executor answer to '{sub}': {ans}\n"
                "Decide the next sub-question or finish."
            )))

        try:
            ai = self.llm.invoke(msgs)
        except Exception as exc:
            return {
                "decision": {"status": "done"},
                "final_answer": "",
                "error": f"Planner call failed: {type(exc).__name__}: {exc}",
                "num_api_calls": state.get("num_api_calls", 0) + 1,
            }

        content = (ai.content or "").strip()
        self._log(f"\nPLANNER RAW: {content}")

        decision = _parse_planner_json(content) or {
            "status": "continue",
            "subquery": "",
        }

        planner_tokens, planner_calls = llm_accounting([ai])

        updates = {
            "decision": decision,
            "total_tokens": state.get("total_tokens", 0) + planner_tokens,
            "num_api_calls": state.get("num_api_calls", 0) + planner_calls,
            "planner_steps": state.get("planner_steps", 0) + 1,
        }

        if decision.get("status") == "done":
            updates["final_answer"] = (decision.get("answer") or "").strip()
        else:
            updates["next_subquery"] = (decision.get("subquery") or "").strip()

        return updates

    def _executor_node(self, state: PEState) -> dict:
        """Run the ReAct sub-agent on the current sub-query; append to scratchpad."""
        sub_query = state.get("next_subquery", "").strip()
        tool_env = self._tool_env  # set in invoke() before running the graph

        if not sub_query:
            ans = "NOT FOUND"
            return {
                "scratchpad": state.get("scratchpad", []) + [
                    ("(empty sub-query)", ans)
                ],
                "num_subqueries": state.get("num_subqueries", 0) + 1,
            }

        tools = tool_env.as_langchain_tools()
        agent = _create_agent(self.llm, tools, system_prompt=EXECUTOR_SYSTEM_PROMPT)

        self._log(f"--> DELEGATING SUB-QUERY: {sub_query}")
        config = {"recursion_limit": self.executor_max_steps * 2 + 1}

        ans = "NOT FOUND"
        exec_tokens = 0
        exec_calls = 0

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=sub_query)]},
                config=config,
            )

            exec_tokens, exec_calls = llm_accounting(result["messages"])

            for m in reversed(result["messages"]):
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                    ans = (m.content or "").strip()
                    break

        except Exception as exc:
            ans = f"[executor error: {type(exc).__name__}: {exc}]"

        self._log(f"<-- EXECUTOR ANSWER: {ans}")
        return {
            "scratchpad": state.get("scratchpad", []) + [(sub_query, ans)],
            "total_tokens": state.get("total_tokens", 0) + exec_tokens,
            "num_api_calls": state.get("num_api_calls", 0) + exec_calls,
            "num_subqueries": state.get("num_subqueries", 0) + 1,
        }

    # --- router ------------------------------------------------------------
    def _route_after_planner(self, state: PEState) -> str:
        """Send to executor unless the planner is done or we've hit the cap."""
        if state.get("error"):
            return "end"
        decision = state.get("decision", {})
        if decision.get("status") == "done":
            return "end"
        if state.get("planner_steps", 0) >= self.max_planner_steps:
            return "end"
        return "executor"

    def _build_graph(self):
        g = StateGraph(PEState)
        g.add_node("planner", self._planner_node)
        g.add_node("executor", self._executor_node)
        g.add_edge(START, "planner")
        g.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"executor": "executor", "end": END},
        )
        g.add_edge("executor", "planner")  # loop back
        return g.compile()

    # --- public interface (matches A1) ------------------------------------
    def invoke(self, question: str, tool_env) -> dict:
        # The executor node needs the per-question tool_env. Graph state can't
        # hold the live object cleanly, so we stash it on self for this run.
        self._tool_env = tool_env

        init: PEState = {
            "question": question,
            "scratchpad": [],
            "total_tokens": 0,
            "num_api_calls": 0,
            "num_subqueries": 0,
            "planner_steps": 0,
        }
        # recursion_limit caps total node visits; each planner+executor round
        # is 2 visits, plus the final planner. Give generous headroom.
        config = {"recursion_limit": self.max_planner_steps * 2 + 5}

        try:
            final_state = self._graph.invoke(init, config=config)
        except Exception as exc:
            return {"answer": "", "total_tokens": 0, "num_api_calls": 0,
                    "error": f"Graph failed: {type(exc).__name__}: {exc}"}
        finally:
            self._tool_env = None

        answer = final_state.get("final_answer", "")
        error = final_state.get("error", "")
        if not answer and not error:
            error = "No final answer produced (planner ended without 'done')"

        out = {
            "answer": answer,
            "total_tokens": final_state.get("total_tokens", 0),
            "num_api_calls": final_state.get("num_api_calls", 0),
            "num_subqueries": final_state.get("num_subqueries", 0),
        }
        if error:
            out["error"] = error
        return out