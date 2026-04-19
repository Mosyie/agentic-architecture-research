from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

# select first 5 rows
ds_small = ds.select(range(5))

# keep only required columns
cols = ["level", "question", "answer", "context", "supporting_facts"]
ds_small = ds_small.remove_columns([c for c in ds.column_names if c not in cols])

# convert to pandas
df = ds_small.to_pandas()

print(df.head())