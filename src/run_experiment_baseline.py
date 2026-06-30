from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.core.metrics             import exact_match, f1_score_text
from src.domains.hotpotqa.loader  import load_hotpotqa_sample, format_prompt
from src.agents.baseline_hotpotqa import BaseLineAgent

def parse_args():
    parser = ArgumentParser(description="Run the HotpotQA baseline agent experiment.")
    parser.add_argument(
        "-d",
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="hard",
        help="HotpotQA difficulty split to run. Default: hard.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    difficulty = args.difficulty

    result_path = Path(f"results/hotpotQa/baseline_{difficulty}.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    print("Loading HotpotQA data...")
    df = load_hotpotqa_sample(difficulty, num_samples=500)

    agent = BaseLineAgent()

    for index, row in df.iterrows():
        print(f"\n--- Question {index + 1} (Difficulty: {row['level']}) ---")
        print(f"Q: {row['question']}")

        prompt = format_prompt(row["question"], row["context"])
        result = agent.invoke(prompt)

        prediction = result.get("answer", "")
        reference = row["answer"]
        tokens_used = result.get("total_tokens", 0)

        em = exact_match(prediction, reference)
        f1 = f1_score_text(prediction, reference)

        results.append(
            {
                "prediction": prediction,
                "reference": reference,
                "f1": f1,
                "em": em,
                "tokens_used": tokens_used,
                "num_api_calls": 1,
            }
        )

        pd.DataFrame(results).to_csv(result_path, index=False)

        print(f"Agent Output: {prediction}")
        print(f"True Answer:  {reference}")
        print(f"EM: {em} | F1: {f1:.3f}")
        print(f"Tokens Used:  {tokens_used}")
        print(f"\nSaved results to: {result_path}")


if __name__ == "__main__":
    main()
