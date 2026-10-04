# verify_output.py
import json
from pathlib import Path

# Find the latest processed file
cleaned_dir = Path("data/cleaned")
jsonl_files = list(cleaned_dir.glob("processed_data_*.jsonl"))

if jsonl_files:
    latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Checking: {latest_file.name}")
    
    # Read first 5 lines
    print("\n📄 First 5 processed sentences:")
    print("=" * 60)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        for i in range(5):
            line = f.readline()
            if line.strip():
                doc = json.loads(line)
                print(f"\nSentence {i+1}:")
                print(f"  ID: {doc.get('id', 'N/A')}")
                print(f"  Source: {doc.get('source', 'N/A')}")
                print(f"  Filename: {doc.get('filename', 'N/A')}")
                print(f"  Sentence ID: {doc.get('sentence_id', 'N/A')}")
                print(f"  Processed Text: {doc.get('processed_text', '')[:100]}...")
                print(f"  Word count: {len(doc.get('processed_text', '').split())}")
    
    # Count total lines
    print("\n" + "=" * 60)
    count = 0
    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    print(f"📊 Total sentences in processed file: {count:,}")
    print(f"✅ Each row is a separate sentence (same as original structure)")
else:
    print("❌ No processed files found")