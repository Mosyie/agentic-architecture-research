from pathlib import Path
import pandas as pd

from src.core.metrics              import exact_match, f1_score_text
from src.domains.hotpotqa.loader   import load_hotpotqa_sample, format_prompt
from src.agents.a1_single_hotpotqa import SingleAgent

RESULT_PATH = Path("results/hotpotQa/single_agent_hard.csv")
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

results = []

def main():
    print("Loading HotpotQA data...")
    df = load_hotpotqa_sample("hard", num_samples=500)

    agent = SingleAgent()

    for index, row in df.iterrows():
        print(f"\n--- Question {index + 1} (Difficulty: {row['level']}) ---")
        print(f"Q: {row['question']}")

        prompt = format_prompt(row['question'], row['context'])
        result = agent.invoke(prompt)

        prediction = result["answer"]
        reference = row["answer"]
        tokens_used = result["total_tokens"]

        em = exact_match(prediction, reference)
        f1 = f1_score_text(prediction, reference)

        results.append({
            "prediction": prediction,
            "reference": reference,
            "f1": f1,
            "em": em,
            "tokens_used": tokens_used,
            "num_api_calls": 1 # default, its the single agent for now
        })

        pd.DataFrame(results).to_csv(RESULT_PATH, index=False)

        print(f"Agent Output: {result['answer']}")
        print(f"True Answer:  {row['answer']}")
        print(f"EM: {em} | F1: {f1:.3f}")
        print(f"Tokens Used:  {result['total_tokens']}")

if __name__ == "__main__":
    main()
