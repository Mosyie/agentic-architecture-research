from pathlib import Path
import pandas as pd

from src.core.metrics                   import exact_match, f1_score_text
from src.domains.hotpotqa.loader        import load_hotpotqa_sample
from src.agents.a1_single_hotpotqa_v2   import SingleAgentV2
from src.domains.hotpotqa.tools         import HotpotQAToolEnvironment


RESULT_PATH = Path("results/hotpotQa/single_agent_v2_hard.csv")
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

results = []


def main():
    print("Loading HotpotQA data...")
    df = load_hotpotqa_sample("hard", num_samples=500)

    agent = SingleAgentV2()

    for index, row in df.iterrows():

        tool_env = HotpotQAToolEnvironment(row["context"])

        print(f"\n--- Question {index + 1} (Difficulty: {row['level']}) ---")
        print(f"Q: {row['question']}")

        result = agent.invoke(
            question=row["question"],
            tool_env=tool_env
        )

        prediction = result.get("answer", "")
        reference = row["answer"]
        tokens_used = result.get("total_tokens", 0)
        num_api_calls = result.get("num_api_calls", 0)
        error = result.get("error", "")

        em = exact_match(prediction, reference)
        f1 = f1_score_text(prediction, reference)

        results.append({
            "prediction": prediction,
            "reference": reference,
            "f1": f1,
            "em": em,
            "tokens_used": tokens_used,
            "num_api_calls": num_api_calls,
            "error": error
        })

        pd.DataFrame(results).to_csv(RESULT_PATH, index=False)

        print(f"Agent Output: {prediction}")
        print(f"True Answer:  {reference}")
        print(f"EM: {em} | F1: {f1:.3f}")
        print(f"Tokens Used: {tokens_used} | API Calls: {num_api_calls}")

        if error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()