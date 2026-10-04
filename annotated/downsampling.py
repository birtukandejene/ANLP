import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\user\Desktop\anlp\data\annotation\final_dataset_expanded.xlsx"
)

OUTPUT_FILE = Path(
    r"C:\Users\user\Desktop\anlp\data\annotation\downsampled_dataset_2000.xlsx"
)

DOMAIN_COLUMN = "domain"

MAX_PER_DOMAIN = 2000

RANDOM_STATE = 42


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DOWNSAMPLING DATASET")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

print("\nOutput file:")
print(OUTPUT_FILE)

print(f"\nMaximum rows per domain: {MAX_PER_DOMAIN:,}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)

print(f"\nTotal rows: {len(df):,}")


# ============================================================
# CHECK DOMAIN COLUMN
# ============================================================

if DOMAIN_COLUMN not in df.columns:

    raise ValueError(
        f"The dataset does not contain the "
        f"'{DOMAIN_COLUMN}' column."
    )


# ============================================================
# ORIGINAL DOMAIN COUNTS
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL DOMAIN DISTRIBUTION")
print("=" * 70)

original_counts = (
    df[DOMAIN_COLUMN]
    .value_counts()
    .sort_index()
)

for domain, count in original_counts.items():

    status = (
        "DOWNSAMPLE"
        if count > MAX_PER_DOMAIN
        else "KEEP AS IS"
    )

    print(
        f"{domain}: "
        f"{count:,} rows "
        f"→ {status}"
    )


# ============================================================
# DOWNSAMPLE
# ============================================================

print("\n" + "=" * 70)
print("DOWNSAMPLING")
print("=" * 70)


downsampled_parts = []


for domain, group in df.groupby(
    DOMAIN_COLUMN,
    dropna=False
):

    current_count = len(group)


    # --------------------------------------------------------
    # If domain has MORE than 2000 rows:
    # randomly select exactly 2000 rows
    # --------------------------------------------------------

    if current_count > MAX_PER_DOMAIN:

        sampled = group.sample(
            n=MAX_PER_DOMAIN,
            random_state=RANDOM_STATE
        )

        print(
            f"{domain}: "
            f"{current_count:,} → "
            f"{MAX_PER_DOMAIN:,}"
        )


    # --------------------------------------------------------
    # If domain has 2000 or fewer rows:
    # keep everything unchanged
    # --------------------------------------------------------

    else:

        sampled = group.copy()

        print(
            f"{domain}: "
            f"{current_count:,} → "
            f"UNCHANGED"
        )


    downsampled_parts.append(sampled)


# ============================================================
# COMBINE ALL DOMAINS
# ============================================================

final = pd.concat(
    downsampled_parts,
    ignore_index=True
)


# ============================================================
# FINAL DOMAIN COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DOMAIN DISTRIBUTION")
print("=" * 70)


final_counts = (
    final[DOMAIN_COLUMN]
    .value_counts()
    .sort_index()
)


# ============================================================
# CREATE REPORT
# ============================================================

report_data = []


all_domains = sorted(
    set(original_counts.index)
)


for domain in all_domains:

    original_count = int(
        original_counts.get(domain, 0)
    )

    final_count = int(
        final_counts.get(domain, 0)
    )

    removed = (
        original_count - final_count
    )


    status = (
        "DOWNSAMPLED"
        if original_count > MAX_PER_DOMAIN
        else "UNCHANGED"
    )


    report_data.append({
        "domain": domain,
        "original_count": original_count,
        "final_count": final_count,
        "rows_removed": removed,
        "status": status
    })


    print(
        f"{domain}: "
        f"original={original_count:,}, "
        f"final={final_count:,}, "
        f"removed={removed:,}"
    )


report_df = pd.DataFrame(
    report_data
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)


over_limit = final_counts[
    final_counts > MAX_PER_DOMAIN
]


if len(over_limit) > 0:

    print(
        "\nERROR: Some domains still exceed "
        f"{MAX_PER_DOMAIN:,} rows."
    )

    print(over_limit)

    raise ValueError(
        "Downsampling validation failed."
    )


print(
    f"\n✓ All domains have "
    f"{MAX_PER_DOMAIN:,} rows or fewer."
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING DATASET")
print("=" * 70)


final.to_excel(
    OUTPUT_FILE,
    index=False
)


print(
    f"\n✓ Downsampled dataset saved to:"
)

print(OUTPUT_FILE)


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_FILE = OUTPUT_FILE.with_name(
    "downsampled_dataset_2000_report.xlsx"
)


with pd.ExcelWriter(
    REPORT_FILE,
    engine="openpyxl"
) as writer:


    report_df.to_excel(
        writer,
        sheet_name="Downsampling Report",
        index=False
    )


    original_counts.rename(
        "original_count"
    ).to_frame().to_excel(
        writer,
        sheet_name="Original Counts"
    )


    final_counts.rename(
        "final_count"
    ).to_frame().to_excel(
        writer,
        sheet_name="Final Counts"
    )


print(
    f"\n✓ Report saved to:"
)

print(REPORT_FILE)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)


print(
    f"\nOriginal total rows: "
    f"{len(df):,}"
)

print(
    f"Final total rows: "
    f"{len(final):,}"
)

print(
    f"Total rows removed: "
    f"{len(df) - len(final):,}"
)

print("\nNo IDs or data values were modified.")

print("=" * 70)

