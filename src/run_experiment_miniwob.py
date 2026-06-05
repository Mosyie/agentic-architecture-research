from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.domains.miniwob.loader import get_tasks
from src.domains.miniwob.env    import make_env, reset_env, format_observation
from src.domains.miniwob.tools  import make_miniwob_tools

from src.agents.a1_single_miniwob_langgraph import SingleAgent


RESULT_DIR = Path("results/miniwob")


def build_agent(architecture: str, verbose: bool = False):
    if architecture == "a1":
        return SingleAgent(verbose=verbose)

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
        choices=["a1"],
        required=True,
        help="Agent architecture to run: a1=single agent.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes (with different seeds) to run per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; episode i uses seed+i.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run the browser headless (default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print the agent's full message trace (LLM turns + tool calls).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result_path = result_path_for(args.architecture, args.difficulty)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = get_tasks(args.difficulty)
    agent = build_agent(args.architecture, verbose=args.verbose)

    print(
        f"Loading MiniWoB++ tasks "
        f"(difficulty={args.difficulty}, tasks={len(tasks)}, episodes={args.episodes})..."
    )

    results = []

    for task in tasks:
        for episode in range(args.episodes):
            seed = args.seed + episode

            print(f"\n--- Task: {task} (episode {episode}, seed {seed}) ---")

            env = None
            try:
                env = make_env(task, headless=args.headless)
                obs, _info = reset_env(env, seed=seed)

                instruction = format_observation(obs)

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

    mean_reward = pd.DataFrame(results)["reward"].mean()
    print(f"\nMean reward: {mean_reward:.3f}")
    print(f"Saved results to: {result_path}")


if __name__ == "__main__":
    main()
