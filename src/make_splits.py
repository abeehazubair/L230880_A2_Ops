"""Create stratified train/val/test split lists (70/15/15) from the audit table.
Run from the project root:  python src/make_splits.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
AUDIT = Path("reports/audit.csv")
OUT = Path("splits")
LABELS = {"RealArt": 0, "AiArtData": 1}    # 1 = AI-generated, the class we want to detect

df = pd.read_csv(AUDIT)
n_all = len(df)
df = df[~df.corrupt]
df = df.sort_values("path").drop_duplicates("sha256", keep="first")   # keep one copy only
print(f"Images: {n_all} -> {len(df)} after removing corrupt files and duplicate copies")
df["label"] = df.cls.map(LABELS)

# 70% train, then split the remaining 30% in half: 15% val, 15% test
train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)

OUT.mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.sort_values("path")[["path", "cls", "label"]].to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:5s} {len(part):4d}  {part.cls.value_counts().to_dict()}  "
          f"AI share={part.label.mean():.1%}")

# Leakage check: no image content (hash) may appear in two splits
h = {n: set(p.sha256) for n, p in [("train", train), ("val", val), ("test", test)]}
assert not (h["train"] & h["val"] or h["train"] & h["test"] or h["val"] & h["test"])
print("Leakage check passed: no image appears in more than one split")
