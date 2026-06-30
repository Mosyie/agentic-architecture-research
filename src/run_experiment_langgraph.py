from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.core.metrics                        import exact_match, f1_score_text
from src.core.run_logger                     import open_run_log
from src.core.llm_client                     import add_llm_callback, remove_llm_callback
from src.domains.hotpotqa.tools              import make_hotpotqa_tools
from src.domains.hotpotqa.loader             import load_hotpotqa_sample

from src.agents.a1_single_hotpotqa_langgraph import SingleAgent
from src.agents.b1_planner_executor_hotpotqa import PlannerExecutorAgent
from src.agents.b2_actor_critic_hotpotqa     import ActorCriticAgent
from src.agents.c1_memory_hotpotqa           import MemoryAgent


NUM_SAMPLES = 500
RESULT_DIR = Path("results/hotpotQa")


def build_agent(architecture: str):
    if architecture == "a1":
        return SingleAgent()

    if architecture == "b1":
        return PlannerExecutorAgent()

    if architecture == 'b2':
        return ActorCriticAgent()

    if architecture == "c1":
        return MemoryAgent()

    raise ValueError(f"Unsupported architecture: {architecture}")


def result_path_for(architecture: str, difficulty: str) -> Path:
    return RESULT_DIR / f"{architecture}_{difficulty}.csv"


def parse_args():
    parser = ArgumentParser(description="Run HotpotQA LangGraph experiments.")
    parser.add_argument(
        "-d",
        "--difficulty",
        default="easy",
        help="HotpotQA difficulty level to load: easy | medium | hard",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        choices=["a1", "b1", "b2", "c1"],
        required=True,
        help="Agent architecture to run: a1=single agent, b1=planner/executor, b2=actor/critic, c1=actor/critic+memory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result_path = result_path_for(args.architecture, args.difficulty)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading HotpotQA data "
        f"(difficulty={args.difficulty}, samples={NUM_SAMPLES})..."
    )

    run_logger = open_run_log(args.architecture, args.difficulty, subdir="hotpotqa")
    add_llm_callback(run_logger)
    print(f"LLM I/O log: {run_logger.path}")
    run_logger.note(
        f"RUN architecture={args.architecture} difficulty={args.difficulty} "
        f"samples={NUM_SAMPLES}"
    )

    df = load_hotpotqa_sample(args.difficulty, num_samples=NUM_SAMPLES)
    agent = build_agent(args.architecture)

    results = []

    try:
        _run_questions(df, agent, result_path, results, run_logger)
    finally:
        remove_llm_callback(run_logger)
        run_logger.close()

    print(f"\nSaved results to: {result_path}")
    print(f"LLM I/O log:     {run_logger.path}")


def _run_questions(df, agent, result_path, results, run_logger):
    for index, row in df.iterrows():
        tools = make_hotpotqa_tools(row["context"])

        print(f"\n--- Question {index + 1} (Difficulty: {row['level']}) ---")
        print(f"Q: {row['question']}")
        run_logger.note(
            f"QUESTION {index + 1} level={row['level']}: {row['question']}"
        )

        result = agent.invoke(
            question=row["question"],
            tools=tools,
        )

        prediction = result.get("answer", "")
        reference = row["answer"]
        tokens_used = result.get("total_tokens", 0)
        num_api_calls = result.get("num_api_calls", 0)
        error = result.get("error", "")

        em = exact_match(prediction, reference)
        f1 = f1_score_text(prediction, reference)

        row_result = {
            "prediction": prediction,
            "reference": reference,
            "f1": f1,
            "em": em,
            "tokens_used": tokens_used,
            "num_api_calls": num_api_calls,
            "error": error,
        }

        results.append(row_result)
        pd.DataFrame(results).to_csv(result_path, index=False)

        print(f"Agent Output: {prediction}")
        print(f"True Answer:  {reference}")
        print(f"EM: {em} | F1: {f1:.3f}")

        usage_line = f"Tokens Used: {tokens_used} | API Calls: {num_api_calls}"
        print(usage_line)

        if error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
