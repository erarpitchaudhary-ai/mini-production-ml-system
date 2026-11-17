
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

INPUT_PATH = Path(
    "data/processed/events_clean.csv.gz"
)

OUTPUT_DIR = Path("data/processed/splits")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run prepare_data.py first."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD CLEANED DATA
# ==========================================

print("\n========== CHRONOLOGICAL SPLITTING ==========")

df = pd.read_csv(INPUT_PATH)

print("Total records:", len(df))

# Verify required columns
required = [
    "timestamp",
    "visitorid",
    "itemid",
    "event"
]

if not all(col in df.columns for col in required):
    raise ValueError("Required columns are missing.")

# ==========================================
# 3. SORT BY TIMESTAMP
# ==========================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# ==========================================
# 4. CALCULATE TEMPORAL CUTOFFS
# ==========================================

train_cutoff, validation_cutoff = np.quantile(
    df["timestamp"].to_numpy(),
    [0.70, 0.85]
)

print("\nTraining cutoff:", int(train_cutoff))
print("Validation cutoff:", int(validation_cutoff))

# ==========================================
# 5. CREATE PARTITIONS
# ==========================================

train = df[
    df["timestamp"] <= train_cutoff
].copy()

validation = df[
    (df["timestamp"] > train_cutoff) &
    (df["timestamp"] <= validation_cutoff)
].copy()

test = df[
    df["timestamp"] > validation_cutoff
].copy()

# ==========================================
# 6. VERIFY TEMPORAL ORDER
# ==========================================

assert train["timestamp"].max() < validation["timestamp"].min()

assert validation["timestamp"].max() < test["timestamp"].min()

assert len(train) + len(validation) + len(test) == len(df)

# ==========================================
# 7. SAVE PARTITIONS
# ==========================================

train.to_csv(
    OUTPUT_DIR / "train.csv.gz",
    index=False,
    compression="gzip"
)

validation.to_csv(
    OUTPUT_DIR / "validation.csv.gz",
    index=False,
    compression="gzip"
)

test.to_csv(
    OUTPUT_DIR / "test.csv.gz",
    index=False,
    compression="gzip"
)

# ==========================================
# 8. GENERATE PARTITION REPORT
# ==========================================

def report_partition(name, data):

    start = pd.to_datetime(
        data["timestamp"].min(),
        unit="ms",
        utc=True
    )

    end = pd.to_datetime(
        data["timestamp"].max(),
        unit="ms",
        utc=True
    )

    print(f"\n{name.upper()}")

    print("Records:", len(data))

    print("Percentage:",
          round(len(data) / len(df) * 100, 2), "%")

    print("Start:", start)
    print("End:", end)

    print("\nEvent distribution:")
    print(data["event"].value_counts())

report_partition("Training", train)
report_partition("Validation", validation)
report_partition("Testing", test)

# ==========================================
# 9. FINAL VERIFICATION
# ==========================================

print("\n========== FINAL VERIFICATION ==========")

print("Training records:", len(train))
print("Validation records:", len(validation))
print("Testing records:", len(test))

print("Temporal leakage check: PASSED")

print("\nSaved partitions:")
print(OUTPUT_DIR)

print("\n========== SPLITTING COMPLETE ==========")
