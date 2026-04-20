from src.domains.hotpotqa.loader import load_hotpotqa_sample, format_prompt
from src.agents.a1_single_hotpotqa import SingleAgent

def main():
    print("Loading HotpotQA data...")
    df = load_hotpotqa_sample("hard", num_samples=5)
    
    agent = SingleAgent()
    
    for index, row in df.iterrows():
        print(f"\n--- Question {index + 1} (Difficulty: {row['level']}) ---")
        print(f"Q: {row['question']}")
        
        prompt = format_prompt(row['question'], row['context'])
        result = agent.invoke(prompt)
        
        print(f"Agent Output: {result['answer']}")
        print(f"True Answer:  {row['answer']}")
        print(f"Tokens Used:  {result['total_tokens']}")

if __name__ == "__main__":
    main()