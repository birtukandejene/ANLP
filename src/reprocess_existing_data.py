# src/reprocess_existing_data.py
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from tqdm import tqdm
import gc

# Use absolute imports (not relative)
from src.data_load import AmharicDataLoader
from src.preprocessing import preprocess_amharic_text

def reprocess_existing_data(input_file: str = None, chunk_size: int = 10000):
    """
    Reprocess existing JSONL dataset - KEEP EACH ROW SEPARATE.
    Replace 'text' with 'processed_text', keep all other fields.
    """
    if input_file is None:
        raw_dir = Path("data/raw")
        jsonl_files = list(raw_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            print("❌ No JSONL files found in data/raw/")
            print("📁 Please place your JSONL file in data/raw/")
            return None
    
    input_file = jsonl_files[0]
    print(f"📂 Found file: {input_file}")
    
    # Initialize loader
    loader = AmharicDataLoader()
    
    # Count total lines
    print("\n📊 Counting total sentences...")
    total_lines = loader.get_total_lines(input_file)
    print(f"📊 Total sentences: {total_lines:,}")
    
    # Output file
    output_dir = Path("data/cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"processed_data_{timestamp}.jsonl"
    
    # Statistics
    total_processed = 0
    total_original_words = 0
    total_processed_words = 0
    chunk_number = 0
    
    print(f"\n🔄 Processing in chunks of {chunk_size} sentences...")
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as out_f:
        
        # Process in chunks
        for chunk in loader.load_jsonl_chunks(input_file, chunk_size):
            chunk_number += 1
            total_processed += len(chunk)
            
            print(f"\n📦 Processing chunk {chunk_number} ({len(chunk)} sentences)...")
            
            # Process each sentence in the chunk
            processed_chunk = []
            for doc in tqdm(chunk, desc=f"Preprocessing chunk {chunk_number}"):
                # Get original text
                original_text = doc.get('text', '')
                
                # Count original words
                original_word_count = len(original_text.split())
                total_original_words += original_word_count
                
                # Preprocess the text
                processed_text = preprocess_amharic_text(
                    original_text,
                    normalize=True,
                    expand_abbr=True,
                    clean=True,
                    segment=False
                )
                
                # Count processed words
                processed_word_count = len(processed_text.split())
                total_processed_words += processed_word_count
                
                # Create new document - REPLACE 'text' WITH 'processed_text'
                new_doc = {
                    'id': doc.get('id', ''),
                    'source': doc.get('source', ''),
                    'filename': doc.get('filename', ''),
                    'sentence_id': doc.get('sentence_id', ''),
                    'processed_text': processed_text
                }
                
                # Keep any other fields that might exist (except 'text')
                for key, value in doc.items():
                    if key not in new_doc and key != 'text':
                        new_doc[key] = value
                
                processed_chunk.append(new_doc)
            
            # Write chunk to output file
            for doc in processed_chunk:
                out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            # Clear memory
            del processed_chunk
            del chunk
            gc.collect()
            
            # Calculate progress
            progress = (total_processed / total_lines) * 100 if total_lines > 0 else 0
            print(f"✅ Chunk {chunk_number} complete. Progress: {progress:.1f}% ({total_processed:,}/{total_lines:,})")
    
    # Calculate statistics
    retention_percentage = (total_processed_words / total_original_words * 100) if total_original_words > 0 else 0
    words_removed = total_original_words - total_processed_words
    
    print(f"\n📊 Processing Complete!")
    print(f"  Total sentences processed: {total_processed:,}")
    print(f"  Original words: {total_original_words:,}")
    print(f"  Processed words: {total_processed_words:,}")
    print(f"  Words removed: {words_removed:,}")
    print(f"  Overall retention: {retention_percentage:.2f}%")
    
    # Get file size
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"\n💾 Saved to: {output_file}")
    print(f"   File size: {file_size:.2f} MB")
    
    # Generate report
    report = {
        'total_sentences': total_processed,
        'total_original_words': total_original_words,
        'total_processed_words': total_processed_words,
        'total_words_removed': words_removed,
        'overall_retention_percentage': retention_percentage,
        'output_file': str(output_file),
        'file_size_mb': file_size
    }
    
    # Save report
    report_file = output_dir / f"processing_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Report saved to: {report_file}")
    
    return report

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Amharic Text Preprocessing - Keep Each Row Separate")
    print("=" * 60)
    
    report = reprocess_existing_data()
    
    if report:
        print("\n" + "=" * 60)
        print("✅ Processing completed successfully!")
        print(f"📁 Output: {report['output_file']}")
        print(f"📊 Total sentences: {report['total_sentences']:,}")
        print(f"💾 File size: {report['file_size_mb']:.2f} MB")
        print("=" * 60)