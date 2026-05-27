#scripts/create_splits.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

ROOT = Path(".")
INPUT = ROOT / "data/interim/cleaned.csv"

OUT_TRAIN = ROOT / "data/interim/train.csv"
OUT_VAL = ROOT / "data/interim/val.csv"
OUT_TEST = ROOT / "data/interim/test.csv"

# load
df = pd.read_csv(INPUT)

# stratify label
df["stratify_label"] = (
    df["event_label"].astype(str)
    + "_"
    + df["info_label"].astype(str)
)

# train/temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["stratify_label"]
)

# val/test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["stratify_label"]
)

# remove helper column
for d in [train_df, val_df, test_df]:
    d.drop(columns=["stratify_label"], inplace=True)

# save
train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL, index=False)
test_df.to_csv(OUT_TEST, index=False)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))