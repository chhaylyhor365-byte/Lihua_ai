"""Download coding datasets from Hugging Face and save one clean CSV for training.

Run:   pip install datasets pandas
       python make_dataset.py
Output: lihua_train.csv  (columns: instruction, response, source)
"""
import pandas as pd
from datasets import load_dataset

# (dataset name, how many rows to take)  <- change the numbers if you want more or less
SOURCES = [
    ("sahil2801/CodeAlpaca-20k", 3000),
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", 4000),
    ("glaiveai/glaive-code-assistant-v3", 4000),
    ("nickrosh/Evol-Instruct-Code-80k-v1", 2000),
    ("iamtarun/code_instructions_120k_alpaca", 2000),
]

OUTPUT_FILE = "lihua_train.csv"
MIN_CHARS = 40       # drop answers that are too short
MAX_CHARS = 4000     # drop answers that are too long


def first(row, names):
    """Return the first non-empty field found in row."""
    for n in names:
        v = row.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def normalize(row):
    instruction = first(row, ["instruction", "question", "prompt", "problem"])
    extra = first(row, ["input"])            # Alpaca style datasets have an extra input field
    response = first(row, ["response", "output", "answer", "solution"])
    if extra and instruction:
        instruction = f"{instruction}\n\n{extra}"
    return instruction, response


def main():
    rows = []
    for name, count in SOURCES:
        print(f"Downloading {name} ...")
        try:
            ds = load_dataset(name, split="train", streaming=True)
            ds = ds.shuffle(seed=42, buffer_size=10000)
            got = 0
            for row in ds:
                instruction, response = normalize(row)
                if not instruction or not (MIN_CHARS <= len(response) <= MAX_CHARS):
                    continue
                rows.append({"instruction": instruction, "response": response, "source": name})
                got += 1
                if got >= count:
                    break
            print(f"   got {got} rows")
        except Exception as e:
            print(f"   SKIPPED {name}: {e}")

    if not rows:
        print("No data downloaded. Check your internet and try again.")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="instruction").sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nDone: {len(df)} rows saved to {OUTPUT_FILE}")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
