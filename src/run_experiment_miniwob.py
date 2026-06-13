from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.domains.miniwob.loader import get_tasks
from src.domains.miniwob.env    import make_env, format_observation
from src.domains.miniwob.tools  import make_miniwob_tools

from src.core.run_logger import open_run_log
from src.core.llm_client import add_llm_callback, remove_llm_callback

from src.agents.a1_single_miniwob_langgraph  import SingleAgent
from src.agents.b1_planner_executor_miniwob  import PlannerExecutorAgent
from src.agents.b2_actor_critic_miniwob      import ActorCriticAgent


RESULT_DIR = Path("results/miniwob")

# Episodes (each a different task variant via seed+i) to run per task. Baked in
# here rather than a flag; bump it for more variants per run.
EPISODES_PER_TASK = 3

# Fixed base seed so every run -- and every architecture -- faces the identical
# task variants (in MiniWoB the reset seed selects the variant). Episode i uses
# BASE_SEED + i.
BASE_SEED = 42


def build_agent(architecture: str):
    if architecture == "a1":
        return SingleAgent()

    if architecture == "b1":
        return PlannerExecutorAgent()

    if architecture == "b2":
        return ActorCriticAgent()

    raise ValueError(f"Unsupported architecture: {architecture}")


def result_path_for(architecture: str, difficulty: str) -> Path:
    return RESULT_DIR / f"{architecture}_{difficulty}.csv"


def parse_args():
    parser = ArgumentParser(description="Run MiniWoB++ LangGraph experiments.")
    parser.add_argument(
        "-d",
        "--difficulty",
        default="easy",
        help="MiniWoB++ difficulty level to load: easy | medium | hard",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        choices=["a1", "b1", "b2"],
        required=True,
        help="Agent architecture to run: a1=single agent, b1=planner/executor, b2=actor/critic.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result_path = result_path_for(args.architecture, args.difficulty)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # One LLM I/O log file per run, registered before the agents are built so
    # every planner/executor/single-agent call is captured.
    run_logger = open_run_log(args.architecture, args.difficulty)
    add_llm_callback(run_logger)

    tasks = get_tasks(args.difficulty)
    agent = build_agent(args.architecture)

    base_seed = BASE_SEED

    print(
        f"Loading MiniWoB++ tasks "
        f"(difficulty={args.difficulty}, tasks={len(tasks)}, "
        f"episodes={EPISODES_PER_TASK}, base_seed={base_seed} -> random task variant)..."
    )
    print(f"LLM I/O log: {run_logger.path}")
    run_logger.note(
        f"RUN architecture={args.architecture} difficulty={args.difficulty} "
        f"base_seed={base_seed} episodes={EPISODES_PER_TASK}"
    )

    results = []

    try:
        _run_tasks(base_seed, tasks, agent, result_path, results, run_logger)
    finally:
        remove_llm_callback(run_logger)
        run_logger.close()

    mean_reward = pd.DataFrame(results)["reward"].mean() if results else 0.0
    print(f"\nMean reward: {mean_reward:.3f}")
    print(f"Saved results to: {result_path}")
    print(f"LLM I/O log:     {run_logger.path}")


def _run_tasks(base_seed, tasks, agent, result_path, results, run_logger):
    for task in tasks:
        for episode in range(EPISODES_PER_TASK):
            seed = base_seed + episode

            print(f"\n--- Task: {task} (episode {episode}, seed {seed}) ---")
            run_logger.note(f"TASK {task}  episode={episode}  seed={seed}")

            env = None
            try:
                env = make_env(task, headless=False)
                obs, _info = env.reset(seed=seed)

                instruction = format_observation(obs, env)

                episode_state = {
                    "last_reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "num_actions": 0,
                }
                tools = make_miniwob_tools(env, episode_state)

                result = agent.invoke(
                    instruction=instruction,
                    tools=tools,
                )

                row_result = {
                    "task": task,
                    "episode": episode,
                    "seed": seed,
                    "reward": episode_state["last_reward"],
                    "num_actions": episode_state["num_actions"],
                    "total_tokens": result.get("total_tokens", 0),
                    "num_api_calls": result.get("num_api_calls", 0),
                    "error": result.get("error", ""),
                }
            except Exception as exc:
                row_result = {
                    "task": task,
                    "episode": episode,
                    "seed": seed,
                    "reward": 0.0,
                    "num_actions": 0,
                    "total_tokens": 0,
                    "num_api_calls": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if env is not None:
                    env.close()

            results.append(row_result)
            pd.DataFrame(results).to_csv(result_path, index=False)

            print(
                f"Reward: {row_result['reward']:.3f} | "
                f"Actions: {row_result['num_actions']} | "
                f"Tokens: {row_result['total_tokens']} | "
                f"API Calls: {row_result['num_api_calls']}"
            )
            if row_result["error"]:
                print(f"Error: {row_result['error']}")


if __name__ == "__main__":
    main()
