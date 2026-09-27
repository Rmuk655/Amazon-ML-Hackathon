import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("dataset")

FILES = {
    "train_source1": DATASET_DIR / "train" / "train_source1.tsv",
    "train_source2": DATASET_DIR / "train" / "train_source2.tsv",
    "train_source3": DATASET_DIR / "train" / "train_source3.tsv",
    "train_ground_truth": DATASET_DIR / "train" / "train_ground_truth.tsv",

    "test_source1": DATASET_DIR / "test" / "test_source1.tsv",
    "test_source2": DATASET_DIR / "test" / "test_source2.tsv",
    "test_source3": DATASET_DIR / "test" / "test_source3.tsv",
}


# Number of examples to show for each column
N_EXAMPLES = 3

# Read large files in chunks so we don't need to load
# the entire 5-million-row file just to inspect nulls.
CHUNK_SIZE = 200_000


# ============================================================
# Analyze one file
# ============================================================

def analyze_file(name, path):

    print("\n" + "=" * 100)
    print(f"FILE: {name}")
    print(f"PATH: {path}")
    print("=" * 100)

    if not path.exists():
        print("ERROR: File not found")
        return

    # --------------------------------------------------------
    # First read a small chunk to get column names
    # --------------------------------------------------------

    first_chunk = pd.read_csv(
        path,
        sep="\t",
        nrows=5
    )

    columns = list(first_chunk.columns)

    print(f"\nColumns: {len(columns)}")
    print("  " + ", ".join(columns))

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    null_counts = {col: 0 for col in columns}

    # Store examples for each column
    # examples[col] = list of rows containing a null in col
    examples = {
        col: []
        for col in columns
    }

    total_rows = 0

    # --------------------------------------------------------
    # Process file in chunks
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        for col in columns:

            null_mask = chunk[col].isna()

            count = null_mask.sum()
            null_counts[col] += count

            # Collect a few examples only
            if count > 0 and len(examples[col]) < N_EXAMPLES:

                remaining = N_EXAMPLES - len(examples[col])

                rows = chunk.loc[null_mask].head(remaining)

                for _, row in rows.iterrows():
                    examples[col].append(row.to_dict())

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(f"\nRows: {total_rows:,}")

    print("\n" + "-" * 100)
    print("MISSING VALUE SUMMARY")
    print("-" * 100)

    print(
        f"{'Column':<25}"
        f"{'Null Count':>15}"
        f"{'Null %':>12}"
    )

    print("-" * 55)

    for col in columns:

        count = null_counts[col]
        percentage = (count / total_rows * 100) if total_rows else 0

        print(
            f"{col:<25}"
            f"{count:>15,}"
            f"{percentage:>11.2f}%"
        )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("NULL VALUE EXAMPLES")
    print("-" * 100)

    found_any = False

    for col in columns:

        if null_counts[col] == 0:
            continue

        found_any = True

        print(f"\n### Column: {col}")
        print(
            f"Showing {len(examples[col])} example(s) "
            f"out of {null_counts[col]:,} null values"
        )

        for i, row in enumerate(examples[col], start=1):

            print(f"\nExample {i}:")

            for field, value in row.items():

                if pd.isna(value):
                    value = "<NULL>"

                print(f"  {field:<20}: {value}")

    if not found_any:
        print("\nNo null values found in this file.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    for name, path in FILES.items():
        analyze_file(name, path)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)