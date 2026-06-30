from datasets import load_dataset
import pandas as pd

def load_hotpotqa_sample(level: str, num_samples=100) -> pd.DataFrame:
    """
    Loads HotpotQA dataset.
    Levels are 'easy', 'medium', or 'hard'.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    
    ds = ds.filter(lambda example: example["level"].lower() == level.lower())
    
    actual_samples = min(num_samples, len(ds))
    ds_small = ds.select(range(actual_samples))
    
    cols = ["level", "question", "answer", "context", "supporting_facts"]
    ds_small = ds_small.remove_columns([c for c in ds.column_names if c not in cols])

    print(f"Loaded {len(ds_small)} samples from HotpotQA with difficulty '{level}'.")
    
    return ds_small.to_pandas()

def format_context(context_dict: dict) -> str:
    formatted = []
    for title, sentences in zip(context_dict["title"], context_dict["sentences"]):
        text = " ".join(sentences)
        formatted.append(f"{title} - {text}")
    return ";\n".join(formatted)

def format_prompt(question: str, context_dict: dict) -> str:
    return question + "\n\nContext:\n" + format_context(context_dict)