# run_preprocess.py
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.reprocess_existing_data import reprocess_existing_data

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