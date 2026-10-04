import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_FILE = Path(
    r"C:\Users\user\Desktop\anlp\data\annotation\balanced_final_dataset.xlsx"
)

ADDITIONAL_FILE = Path(
    r"C:\Users\user\Desktop\anlp\annotated\merged_annotated_dataset_filled.xlsx"
)

OUTPUT_FILE = Path(
    r"C:\Users\user\Desktop\anlp\data\annotation\final_dataset_expanded.xlsx"
)

ID_COLUMN = "id"
DOMAIN_COLUMN = "domain"


# ============================================================
# TARGET DOMAINS TO INCREASE
# ============================================================

TARGET_DOMAINS = [
    "ታሪክ",
    "ጤና",
    "ባህል",
    "ቴክኖሎጂ",
    "መገናኛ ብዙሃን",
    "ሥነ ጽሑፍ",
    "ትምህርት",
    "ግብርና",
    "ትራንስፖርት",
    "መዝናኛ",
    "ፍልስፍና",
    "ሳይንስ"
]


# ============================================================
# HELPER: NORMALIZE ID FOR COMPARISON ONLY
# ============================================================

def normalize_id(value):
    """
    Normalize IDs ONLY for duplicate checking.

    IMPORTANT:
    The original 'id' column is NEVER changed.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Treat Excel numeric IDs like 123.0 as 123
    if value.endswith(".0"):
        try:
            value = str(int(float(value)))
        except ValueError:
            pass

    return value


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EXPANDING SELECTED DOMAINS")
print("=" * 70)

print("\nBase dataset:")
print(BASE_FILE)

print("\nAdditional dataset:")
print(ADDITIONAL_FILE)

print("\nTarget domains:")

for domain in TARGET_DOMAINS:
    print(f"  - {domain}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

base = pd.read_excel(BASE_FILE)
additional = pd.read_excel(ADDITIONAL_FILE)

print(f"\nBase rows:       {len(base):,}")
print(f"Additional rows: {len(additional):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING REQUIRED COLUMNS")
print("=" * 70)

for df_name, df in [
    ("Base dataset", base),
    ("Additional dataset", additional)
]:

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"{df_name} does not contain '{ID_COLUMN}' column."
        )

    if DOMAIN_COLUMN not in df.columns:
        raise ValueError(
            f"{df_name} does not contain '{DOMAIN_COLUMN}' column."
        )

print("✓ Required columns found.")


# ============================================================
# CREATE INTERNAL NORMALIZED IDs
#
# IMPORTANT:
# This DOES NOT MODIFY THE ORIGINAL ID COLUMN.
# ============================================================

base["_comparison_id"] = (
    base[ID_COLUMN].apply(normalize_id)
)

additional["_comparison_id"] = (
    additional[ID_COLUMN].apply(normalize_id)
)


# ============================================================
# CHECK DUPLICATES IN BASE DATASET
# ============================================================

print("\n" + "=" * 70)
print("CHECKING BASE DATASET DUPLICATES")
print("=" * 70)

base_duplicate_mask = (
    base["_comparison_id"].notna()
    &
    base["_comparison_id"].duplicated(keep=False)
)

base_duplicate_count = base_duplicate_mask.sum()

print(
    f"\nRows with duplicate IDs in base: "
    f"{base_duplicate_count:,}"
)

if base_duplicate_count > 0:

    print(
        "\nWARNING:"
        " Duplicate IDs already exist in the base dataset."
    )

    print(
        "These rows will NOT be modified or removed."
    )

else:

    print(
        "✓ No duplicate IDs in base dataset."
    )


# ============================================================
# FILTER ADDITIONAL DATA
# ONLY SELECT TARGET DOMAINS
# ============================================================

print("\n" + "=" * 70)
print("FILTERING TARGET DOMAINS")
print("=" * 70)

additional_target = additional[
    additional[DOMAIN_COLUMN].isin(TARGET_DOMAINS)
].copy()

print(
    f"\nAdditional rows in target domains: "
    f"{len(additional_target):,}"
)


# ============================================================
# CHECK WHICH TARGET DOMAINS EXIST
# ============================================================

print("\nTarget domain availability:")

for domain in TARGET_DOMAINS:

    count = (
        additional_target[
            additional_target[DOMAIN_COLUMN] == domain
        ].shape[0]
    )

    print(
        f"{domain}: {count:,} rows"
    )


# ============================================================
# REMOVE ROWS WITH MISSING IDs
#
# Original data is NOT modified.
# Only rows without an ID are excluded from additions
# because duplicate checking is impossible.
# ============================================================

print("\n" + "=" * 70)
print("CHECKING MISSING IDs")
print("=" * 70)

before_missing_check = len(additional_target)

additional_target = additional_target[
    additional_target["_comparison_id"].notna()
].copy()

removed_missing_ids = (
    before_missing_check - len(additional_target)
)

print(
    f"Rows removed because ID is missing: "
    f"{removed_missing_ids:,}"
)


# ============================================================
# REMOVE DUPLICATE IDs INSIDE ADDITIONAL DATA
#
# Keep first occurrence.
# ============================================================

print("\n" + "=" * 70)
print("REMOVING DUPLICATES INSIDE ADDITIONAL DATA")
print("=" * 70)

before_deduplication = len(additional_target)

additional_target = additional_target.drop_duplicates(
    subset="_comparison_id",
    keep="first"
).copy()

removed_internal_duplicates = (
    before_deduplication - len(additional_target)
)

print(
    f"Duplicate rows removed from additional data: "
    f"{removed_internal_duplicates:,}"
)


# ============================================================
# CREATE SET OF BASE IDs
# ============================================================

base_ids = set(
    base["_comparison_id"]
    .dropna()
)

print(
    f"\nUnique IDs in base dataset: "
    f"{len(base_ids):,}"
)


# ============================================================
# REMOVE ADDITIONAL ROWS
# WHOSE ID ALREADY EXISTS IN BASE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ID OVERLAP")
print("=" * 70)

already_in_base_mask = (
    additional_target["_comparison_id"]
    .isin(base_ids)
)

already_in_base_count = (
    already_in_base_mask.sum()
)

print(
    f"\nAdditional rows already in base: "
    f"{already_in_base_count:,}"
)


# ============================================================
# KEEP ONLY NEW UNIQUE ROWS
# ============================================================

additions = additional_target[
    ~already_in_base_mask
].copy()

print(
    f"New unique rows available to add: "
    f"{len(additions):,}"
)


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL ID SAFETY CHECK")
print("=" * 70)

addition_ids = set(
    additions["_comparison_id"]
    .dropna()
)

overlap = addition_ids.intersection(
    base_ids
)

if overlap:

    print(
        f"\nERROR: {len(overlap):,} IDs overlap!"
    )

    print("\nExample overlapping IDs:")
    print(list(overlap)[:20])

    raise ValueError(
        "STOP: Additional data contains IDs "
        "already existing in the base dataset."
    )

print(
    "✓ No additional ID already exists in base."
)


# ============================================================
# CHECK DUPLICATES INSIDE ADDITIONS
# ============================================================

duplicate_additions = (
    additions["_comparison_id"]
    .duplicated()
    .sum()
)

if duplicate_additions > 0:

    raise ValueError(
        f"STOP: {duplicate_additions:,} duplicate "
        "IDs exist inside additions."
    )

print(
    "✓ No duplicate IDs inside additions."
)


# ============================================================
# DOMAIN COUNTS BEFORE MERGING
# ============================================================

print("\n" + "=" * 70)
print("DOMAIN COUNTS BEFORE MERGING")
print("=" * 70)

base_domain_counts = (
    base[DOMAIN_COLUMN]
    .value_counts()
)

for domain in TARGET_DOMAINS:

    count = int(
        base_domain_counts.get(domain, 0)
    )

    percentage = (
        count / len(base) * 100
        if len(base) > 0
        else 0
    )

    print(
        f"{domain}: "
        f"{count:,} rows "
        f"({percentage:.2f}%)"
    )


# ============================================================
# COUNT ADDED ROWS PER DOMAIN
# ============================================================

added_domain_counts = (
    additions[DOMAIN_COLUMN]
    .value_counts()
)


# ============================================================
# MERGE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("MERGING DATASETS")
print("=" * 70)

# Remove helper column before merging.
# The original ID values remain COMPLETELY unchanged.

base_final = base.drop(
    columns=["_comparison_id"],
    errors="ignore"
)

additions_final = additions.drop(
    columns=["_comparison_id"],
    errors="ignore"
)


# Combine base + new rows

final = pd.concat(
    [
        base_final,
        additions_final
    ],
    ignore_index=True
)

print(
    f"\nRows before merging: "
    f"{len(base_final):,}"
)

print(
    f"Rows added: "
    f"{len(additions_final):,}"
)

print(
    f"Final rows: "
    f"{len(final):,}"
)


# ============================================================
# FINAL DUPLICATE CHECK
#
# Create temporary comparison IDs only for validation.
# Original ID column is NOT modified.
# ============================================================

final["_comparison_id"] = (
    final[ID_COLUMN].apply(normalize_id)
)

final_duplicate_mask = (
    final["_comparison_id"].notna()
    &
    final["_comparison_id"].duplicated(
        keep=False
    )
)

final_duplicate_count = (
    final_duplicate_mask.sum()
)


print("\n" + "=" * 70)
print("FINAL DUPLICATE VALIDATION")
print("=" * 70)

print(
    f"\nRows with duplicate IDs: "
    f"{final_duplicate_count:,}"
)

if final_duplicate_count > 0:

    print(
        "\nWARNING:"
        " Duplicate IDs exist in the final dataset."
    )

    print(
        "These duplicates existed in the original "
        "base dataset."
    )

else:

    print(
        "✓ Final dataset has ZERO duplicate IDs."
    )


# Remove temporary helper column

final = final.drop(
    columns=["_comparison_id"],
    errors="ignore"
)


# ============================================================
# FINAL DOMAIN COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DOMAIN DISTRIBUTION")
print("=" * 70)

final_domain_counts = (
    final[DOMAIN_COLUMN]
    .value_counts()
)


# ============================================================
# CREATE DOMAIN REPORT
# ============================================================

report_data = []

for domain in TARGET_DOMAINS:

    before_count = int(
        base_domain_counts.get(domain, 0)
    )

    added_count = int(
        added_domain_counts.get(domain, 0)
    )

    final_count = int(
        final_domain_counts.get(domain, 0)
    )

    before_percentage = (
        before_count / len(base_final) * 100
        if len(base_final) > 0
        else 0
    )

    final_percentage = (
        final_count / len(final) * 100
        if len(final) > 0
        else 0
    )

    report_data.append({
        "domain": domain,
        "before_count": before_count,
        "before_percentage": round(
            before_percentage,
            2
        ),
        "added": added_count,
        "final_count": final_count,
        "final_percentage": round(
            final_percentage,
            2
        )
    })

    print(
        f"{domain}: "
        f"before={before_count:,}, "
        f"added={added_count:,}, "
        f"final={final_count:,}, "
        f"percentage={final_percentage:.2f}%"
    )


report_df = pd.DataFrame(
    report_data
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL DATASET")
print("=" * 70)

final.to_excel(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nFinal dataset saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_FILE = OUTPUT_FILE.with_name(
    "final_dataset_expansion_report.xlsx"
)


with pd.ExcelWriter(
    REPORT_FILE,
    engine="openpyxl"
) as writer:

    report_df.to_excel(
        writer,
        sheet_name="Target Domain Report",
        index=False
    )

    final_domain_counts.rename(
        "count"
    ).to_frame().to_excel(
        writer,
        sheet_name="All Final Domains"
    )


print(
    f"\nReport saved to:\n"
    f"{REPORT_FILE}"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nBase rows: "
    f"{len(base_final):,}"
)

print(
    f"Rows added: "
    f"{len(additions_final):,}"
)

print(
    f"Final rows: "
    f"{len(final):,}"
)

print("\nOriginal IDs were NOT modified.")

print("=" * 70)