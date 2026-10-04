# create_balanced_annotation.py
import json
import random
from pathlib import Path
from collections import defaultdict
import glob

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule


# ============================================================
# CONFIGURATION
# ============================================================

# Find the latest processed CSV file
csv_files = glob.glob("data/cleaned/processed_data_*.csv")

if csv_files:
    # Get the most recent file
    INPUT_FILE = Path(max(csv_files, key=lambda x: Path(x).stat().st_mtime))
else:
    # Try JSONL
    jsonl_files = glob.glob("data/cleaned/processed_data_*.jsonl")
    if jsonl_files:
        INPUT_FILE = Path(max(jsonl_files, key=lambda x: Path(x).stat().st_mtime))
    else:
        raise FileNotFoundError("No processed data found in data/cleaned/")

OUTPUT_DIR = Path("data/annotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "amharic_annotation_balanced.xlsx"

# Target number of sentences
TARGET_SAMPLE = 50000

# Fixed seed for reproducibility
RANDOM_SEED = 42


# ============================================================
# DOMAINS IN AMHARIC
# ============================================================

DOMAINS_AMHARIC = [
    "ፖለቲካ",          # Politics
    "ታሪክ",            # History
    "ሃይማኖት",         # Religion
    "ፍልስፍና",         # Philosophy
    "ትምህርት",         # Education
    "ሳይንስ",          # Science
    "ቴክኖሎጂ",        # Technology
    "ግብርና",          # Agriculture
    "ጤና",             # Health
    "ህግ",             # Law
    "ኢኮኖሚ",          # Economics
    "ስነ-ጽሁፍ",        # Literature
    "ባህል",            # Culture
    "አካባቢ",          # Environment
    "ንግድ",            # Business
    "ስፖርት",          # Sports
    "ሌላ",             # Other
]

# English mapping for reference
DOMAINS_ENGLISH = {
    "ፖለቲካ": "Politics",
    "ታሪክ": "History",
    "ሃይማኖት": "Religion",
    "ፍልስፍና": "Philosophy",
    "ትምህርት": "Education",
    "ሳይንስ": "Science",
    "ቴክኖሎጂ": "Technology",
    "ግብርና": "Agriculture",
    "ጤና": "Health",
    "ህግ": "Law",
    "ኢኮኖሚ": "Economics",
    "ስነ-ጽሁፍ": "Literature",
    "ባህል": "Culture",
    "አካባቢ": "Environment",
    "ንግድ": "Business",
    "ስፖርት": "Sports",
    "ሌላ": "Other",
}

STATUS_OPTIONS = [
    "pending",
    "annotated",
    "reviewed",
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PREPARING BALANCED AMHARIC ANNOTATION WORKBOOK")
print("=" * 60)

print(f"\n📂 Input file: {INPUT_FILE}")

# Load as CSV
try:
    import pandas as pd
    df = pd.read_csv(INPUT_FILE)
    print(f"📊 Loaded {len(df):,} records from CSV")
    
    # Convert to list of dicts
    records = df.to_dict('records')
    
except Exception as e:
    print(f"⚠️ Could not load as CSV: {e}")
    print("📖 Trying to load as JSONL...")
    
    # Fallback to JSONL
    records = []
    bad_json = 0
    empty_text = 0
    
    with INPUT_FILE.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            
            # Check if we have processed_text or text
            text = str(record.get("processed_text", record.get("text", ""))).strip()
            if not text:
                empty_text += 1
                continue
            
            records.append(record)
    
    print(f"✅ Loaded {len(records):,} valid records")
    if bad_json > 0:
        print(f"⚠️ Skipped {bad_json:,} bad JSON lines")
    if empty_text > 0:
        print(f"⚠️ Skipped {empty_text:,} empty text records")


# ============================================================
# CHECK DATA SIZE
# ============================================================

if len(records) < TARGET_SAMPLE:
    print(f"\n⚠️ Warning: Only {len(records):,} records available. Using all records.")
    TARGET_SAMPLE = len(records)


# ============================================================
# BALANCED SAMPLING BY DOMAIN
# ============================================================

print(f"\n🎯 Target: {TARGET_SAMPLE:,} sentences")
print("📊 Creating balanced sample by domain...")

# Group records by domain
domain_groups = defaultdict(list)

for record in records:
    # Try to get domain from the record
    domain = record.get('domain', 'ሌላ')  # Default to 'Other'
    
    # If domain is in English, convert to Amharic
    if domain in DOMAINS_ENGLISH.values():
        for amh, eng in DOMAINS_ENGLISH.items():
            if eng == domain:
                domain = amh
                break
    
    # If domain not in our list, assign to 'ሌላ'
    if domain not in DOMAINS_AMHARIC:
        domain = 'ሌላ'
    
    domain_groups[domain].append(record)

# Count domains
print("\n📊 Domain distribution in source data:")
for domain in sorted(domain_groups.keys(), key=lambda x: len(domain_groups[x]), reverse=True):
    count = len(domain_groups[domain])
    print(f"  {domain}: {count:,} ({count/len(records)*100:.1f}%)")

# Calculate how many to sample from each domain
valid_domains = [d for d in DOMAINS_AMHARIC if d in domain_groups and len(domain_groups[d]) > 0]
samples_per_domain = TARGET_SAMPLE // len(valid_domains)

print(f"\n📊 Sampling {samples_per_domain:,} per domain from {len(valid_domains)} domains")

# Sample from each domain
sampled_records = []
sampling_stats = []

for domain in valid_domains:
    domain_records = domain_groups[domain]
    n_available = len(domain_records)
    n_sample = min(samples_per_domain, n_available)
    
    # Random sample
    random.seed(RANDOM_SEED)
    sample = random.sample(domain_records, n_sample)
    
    sampled_records.extend(sample)
    sampling_stats.append({
        'domain': domain,
        'available': n_available,
        'sampled': n_sample,
        'percentage': (n_sample / n_available * 100) if n_available > 0 else 0
    })

# If we still need more records, take from largest domains
if len(sampled_records) < TARGET_SAMPLE:
    remaining_needed = TARGET_SAMPLE - len(sampled_records)
    print(f"\n⚠️ Need {remaining_needed} more records. Sampling from largest domains...")
    
    # Get domains sorted by size (largest first)
    sorted_domains = sorted(domain_groups.keys(), key=lambda x: len(domain_groups[x]), reverse=True)
    
    for domain in sorted_domains:
        if remaining_needed <= 0:
            break
        
        # Get records not already sampled
        existing_ids = {r.get('id', '') for r in sampled_records}
        available = [r for r in domain_groups[domain] if r.get('id', '') not in existing_ids]
        
        if available:
            n_sample = min(remaining_needed, len(available))
            random.seed(RANDOM_SEED)
            extra = random.sample(available, n_sample)
            sampled_records.extend(extra)
            remaining_needed -= n_sample

# Shuffle the final sample
random.seed(RANDOM_SEED)
random.shuffle(sampled_records)

print(f"\n✅ Final sample: {len(sampled_records):,} sentences")

# Show sampling stats
print("\n📊 Sampling results:")
for stat in sampling_stats:
    print(f"  {stat['domain']}: {stat['sampled']:,}/{stat['available']:,} ({stat['percentage']:.1f}%)")


# ============================================================
# CREATE WORKBOOK
# ============================================================

wb = Workbook()
ws = wb.active
ws.title = "Annotation"


# ============================================================
# COLUMN HEADERS (Amharic labels for annotators)
# ============================================================

headers = [
    "ቁጥር",              # annotation_id
    "መታወቂያ",          # id
    "ምንጭ",             # source
    "ጎራ",              # domain
    "ጽሑፍ",              # text
    "ቁልፍ ቃላት",        # keywords
    "ርዕሰ ጉዳይ",        # topics
    "ማስታወሻ",          # notes
    "አስተያየት ሰጪ",     # annotator
    "ገምጋሚ",           # reviewer
    "ሁኔታ",             # status
]

ws.append(headers)


# ============================================================
# HEADER STYLE
# ============================================================

header_fill = PatternFill(
    fill_type="solid",
    fgColor="1F4E78"
)

header_font = Font(
    bold=True,
    color="FFFFFF",
    size=11
)

header_alignment = Alignment(
    horizontal="center",
    vertical="center",
    wrap_text=True
)

thin_border = Border(
    left=Side(style="thin", color="D9E1F2"),
    right=Side(style="thin", color="D9E1F2"),
    top=Side(style="thin", color="D9E1F2"),
    bottom=Side(style="thin", color="D9E1F2"),
)

for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = header_alignment
    cell.border = thin_border

ws.row_dimensions[1].height = 30


# ============================================================
# ADD RECORDS
# ============================================================

for annotation_number, record in enumerate(sampled_records, start=1):
    # Get the processed text (use 'processed_text' if available, else 'text')
    text = record.get('processed_text', record.get('text', ''))
    
    # Get domain (already in Amharic)
    domain = record.get('domain', 'ሌላ')
    if domain not in DOMAINS_AMHARIC:
        domain = 'ሌላ'
    
    row = [
        annotation_number,           # ቁጥር
        record.get('id', ''),        # መታወቂያ
        record.get('source', ''),    # ምንጭ
        domain,                       # ጎራ
        text,                         # ጽሑፍ
        '',                          # ቁልፍ ቃላት (to be filled)
        '',                          # ርዕሰ ጉዳይ (to be filled)
        '',                          # ማስታወሻ (to be filled)
        '',                          # አስተያየት ሰጪ (to be filled)
        '',                          # ገምጋሚ (to be filled)
        'pending',                   # ሁኔታ
    ]
    
    ws.append(row)


# ============================================================
# FORMAT DATA CELLS
# ============================================================

for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    for cell in row:
        cell.alignment = Alignment(
            vertical="top",
            horizontal="left",
            wrap_text=True
        )
        cell.border = thin_border


# ============================================================
# COLUMN WIDTHS
# ============================================================

column_widths = {
    "A": 10,   # ቁጥር
    "B": 38,   # መታወቂያ
    "C": 28,   # ምንጭ
    "D": 22,   # ጎራ
    "E": 90,   # ጽሑፍ
    "F": 40,   # ቁልፍ ቃላት
    "G": 35,   # ርዕሰ ጉዳይ
    "H": 35,   # ማስታወሻ
    "I": 18,   # አስተያየት ሰጪ
    "J": 18,   # ገምጋሚ
    "K": 15,   # ሁኔታ
}

for column, width in column_widths.items():
    ws.column_dimensions[column].width = width


# ============================================================
# ROW HEIGHTS
# ============================================================

for row_number in range(2, ws.max_row + 1):
    ws.row_dimensions[row_number].height = 85


# ============================================================
# FREEZE HEADER & FILTER
# ============================================================

ws.freeze_panes = "A2"
ws.auto_filter.ref = f"A1:K{ws.max_row}"


# ============================================================
# DATA VALIDATION — DOMAIN (Amharic)
# ============================================================

domain_formula = '"' + ",".join(DOMAINS_AMHARIC) + '"'

domain_validation = DataValidation(
    type="list",
    formula1=domain_formula,
    allow_blank=True
)

domain_validation.error = "እባክዎ ከተዘረዘሩት መካከል ተገቢውን ጎራ ይምረጡ።"
domain_validation.errorTitle = "ልክ ያልሆነ ጎራ"
domain_validation.prompt = "ጽሑፉን በተሻለ ሁኔታ የሚገልጸውን ጎራ ይምረጡ።"
domain_validation.promptTitle = "ጎራ"

ws.add_data_validation(domain_validation)
domain_validation.add(f"D2:D{ws.max_row}")


# ============================================================
# DATA VALIDATION — STATUS
# ============================================================

status_formula = '"' + ",".join(STATUS_OPTIONS) + '"'

status_validation = DataValidation(
    type="list",
    formula1=status_formula,
    allow_blank=False
)

status_validation.error = "እባክዎ ትክክለኛ ሁኔታ ይምረጡ።"
status_validation.errorTitle = "ልክ ያልሆነ ሁኔታ"

ws.add_data_validation(status_validation)
status_validation.add(f"K2:K{ws.max_row}")


# ============================================================
# CONDITIONAL FORMATTING — STATUS
# ============================================================

ws.conditional_formatting.add(
    f"K2:K{ws.max_row}",
    CellIsRule(
        operator="equal",
        formula=['"pending"'],
        fill=PatternFill(fill_type="solid", fgColor="FFF2CC")
    )
)

ws.conditional_formatting.add(
    f"K2:K{ws.max_row}",
    CellIsRule(
        operator="equal",
        formula=['"annotated"'],
        fill=PatternFill(fill_type="solid", fgColor="D9EAD3")
    )
)

ws.conditional_formatting.add(
    f"K2:K{ws.max_row}",
    CellIsRule(
        operator="equal",
        formula=['"reviewed"'],
        fill=PatternFill(fill_type="solid", fgColor="CFE2F3")
    )
)


# ============================================================
# SECOND SHEET — ANNOTATION GUIDELINES (Amharic)
# ============================================================

guide = wb.create_sheet("የአስተያየት መመሪያ")

guidelines = [
    ("ዓላማ", "ለአማርኛ ቁልፍ ቃላት እና ርዕሰ ጉዳይ ማውጣት ጥራት ያለው መረጃ ለመፍጠር።"),
    ("ጎራ", "የጽሑፉን ዋና ርዕሰ ጉዳይ በሚገልጽ መልኩ ይምረጡ።"),
    ("ቁልፍ ቃላት", "ከጽሑፉ ውስጥ እስከ 5 አስፈላጊ ቁልፍ ቃላት ወይም ሀረጎች ይምረጡ።"),
    ("ቁልፍ ቃል ህግ", "በጽሑፉ ውስጥ የሌሉ ቃላትን አይፈጥሩ።"),
    ("ሀረጎች", "ትርጉም ያለው ሀረግ ከሆነ ብዙ ቃላት ያሉት ቁልፍ ቃል ሊሆን ይችላል።"),
    ("ርዕሰ ጉዳይ", "ጽሑፉ በዋነኛነት ስለምን እንደሆነ የሚገልጽ ሰፊ ምድብ ይስጡ።"),
    ("ቁልፍ ቃል vs ርዕሰ ጉዳይ", "ቁልፍ ቃላት ከጽሑፉ የተወሰዱ ሲሆኑ ርዕሰ ጉዳይ ደግሞ አጠቃላይ ምድብ ነው።"),
    ("ማስታወሻ", "ውሳኔ ማብራሪያ በሚያስፈልግ ጊዜ ብቻ ይጠቀሙ።"),
    ("አስተያየት ሰጪ", "የአስተያየት ሰጪውን ስም ወይም መለያ ያስገቡ።"),
    ("ገምጋሚ", "ከግምገማ በኋላ የገምጋሚውን ስም ያስገቡ።"),
    ("ሁኔታ", "ከአስተያየት በፊት pending፣ ከአስተያየት በኋላ annotated፣ ከግምገማ በኋላ reviewed ይጠቀሙ።"),
    ("አስፈላጊ", "በአስተያየት ጊዜ የመጀመሪያውን ጽሑፍ አይቀይሩ።"),
    ("ዒላማ", "ፕሮጀክቱ በተመረጡት 5 ቁልፍ ቃላት ላይ ከፍተኛ መግባባት ላይ ለመድረስ ያለመ ነው።"),
]

# Guidelines header
guide.append(["ዝርዝር", "መመሪያ"])
for cell in guide[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = header_alignment
    cell.border = thin_border

# Add guidelines
for item, explanation in guidelines:
    guide.append([item, explanation])

# Format guidelines
for row in guide.iter_rows(min_row=2, max_row=guide.max_row):
    for cell in row:
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        cell.border = thin_border

guide.column_dimensions["A"].width = 25
guide.column_dimensions["B"].width = 100
guide.freeze_panes = "A2"

for row_number in range(2, guide.max_row + 1):
    guide.row_dimensions[row_number].height = 45


# ============================================================
# THIRD SHEET — DOMAIN LIST (Amharic)
# ============================================================

domain_sheet = wb.create_sheet("የጎራ ዝርዝር")

domain_sheet["A1"] = "የሚገኙ ጎራዎች"
domain_sheet["A1"].fill = header_fill
domain_sheet["A1"].font = header_font
domain_sheet["A1"].alignment = header_alignment

for index, domain in enumerate(DOMAINS_AMHARIC, start=2):
    domain_sheet.cell(row=index, column=1, value=domain)

domain_sheet.column_dimensions["A"].width = 30


# ============================================================
# FOURTH SHEET — DOMAIN MAPPING (Reference)
# ============================================================

mapping_sheet = wb.create_sheet("Domain Mapping")

mapping_sheet["A1"] = "አማርኛ"
mapping_sheet["B1"] = "English"
mapping_sheet["C1"] = "Description"

for cell in mapping_sheet[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = header_alignment
    cell.border = thin_border

domain_descriptions = {
    "ፖለቲካ": "Politics - Political news, government, elections",
    "ታሪክ": "History - Historical events, figures, periods",
    "ሃይማኖት": "Religion - Faith, worship, spirituality, religious practices",
    "ፍልስፍና": "Philosophy - Philosophical ideas, thinkers, concepts",
    "ትምህርት": "Education - Schools, learning, teaching, education system",
    "ሳይንስ": "Science - Scientific discoveries, research, studies",
    "ቴክኖሎጂ": "Technology - Tech, digital, innovation, computers",
    "ግብርና": "Agriculture - Farming, crops, livestock, rural development",
    "ጤና": "Health - Medical, healthcare, diseases, wellness",
    "ህግ": "Law - Legal matters, courts, justice, legislation",
    "ኢኮኖሚ": "Economics - Economy, finance, business, markets",
    "ስነ-ጽሁፍ": "Literature - Books, authors, poetry, writing",
    "ባህል": "Culture - Traditions, customs, arts, heritage",
    "አካባቢ": "Environment - Nature, conservation, climate, ecology",
    "ንግድ": "Business - Commerce, trade, companies, entrepreneurship",
    "ስፖርት": "Sports - Athletics, games, teams, competitions",
    "ሌላ": "Other - Topics not covered in other categories",
}

for idx, (amharic, english) in enumerate(DOMAINS_ENGLISH.items(), start=2):
    description = domain_descriptions.get(amharic, "")
    mapping_sheet.cell(row=idx, column=1, value=amharic)
    mapping_sheet.cell(row=idx, column=2, value=english)
    mapping_sheet.cell(row=idx, column=3, value=description)

mapping_sheet.column_dimensions["A"].width = 25
mapping_sheet.column_dimensions["B"].width = 25
mapping_sheet.column_dimensions["C"].width = 70


# ============================================================
# SAVE WORKBOOK
# ============================================================

print("\n💾 Saving workbook...")
wb.save(OUTPUT_FILE)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("✅ BALANCED ANNOTATION WORKBOOK CREATED")
print("=" * 60)

print(f"\n📊 Records included : {len(sampled_records):,}")
print(f"📁 Output file      : {OUTPUT_FILE}")

print("\n📋 Workbook sheets:")
print("  1. Annotation (ዋና ሰነድ)")
print("  2. የአስተያየት መመሪያ (Guidelines)")
print("  3. የጎራ ዝርዝር (Domains)")
print("  4. Domain Mapping (Reference)")

print("\n📝 Annotation Columns (Amharic):")
print("  - ቁጥር (annotation_id)")
print("  - መታወቂያ (id)")
print("  - ምንጭ (source)")
print("  - ጎራ (domain) ← Dropdown in Amharic")
print("  - ጽሑፍ (text)")
print("  - ቁልፍ ቃላት (keywords) ← To annotate")
print("  - ርዕሰ ጉዳይ (topics) ← To annotate")
print("  - ማስታወሻ (notes)")
print("  - አስተያየት ሰጪ (annotator)")
print("  - ገምጋሚ (reviewer)")
print("  - ሁኔታ (status) ← Dropdown")

print("\n✅ Features:")
print("  - ✅ Balanced by domain")
print("  - ✅ Domain names in Amharic")
print("  - ✅ 50,000 sentences target")
print("  - ✅ Original text preserved")
print("  - ✅ Dropdowns for domain and status")
print("  - ✅ Color-coded status")
print("  - ✅ Text wrapping enabled")
print("  - ✅ Header frozen")
print("  - ✅ Filters enabled")
print("  - ✅ Amharic guidelines included")

print("\n📊 Domain Distribution in Sample:")
for domain in DOMAINS_AMHARIC:
    count = sum(1 for r in sampled_records if r.get('domain', 'ሌላ') == domain)
    if count > 0:
        print(f"  {domain}: {count:,} ({count/len(sampled_records)*100:.1f}%)")

print("\n🎯 Next Steps:")
print("  1. Open the Excel file")
print("  2. Annotate keywords and topics")
print("  3. Use status dropdown to track progress")
print("  4. Export annotations when complete")

print("\nDone.")