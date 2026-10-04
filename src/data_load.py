# src/data_loader.py
import json
from pathlib import Path
from typing import List, Dict, Generator
import re

class AmharicDataLoader:
    """Load and process existing Amharic JSONL datasets with chunking"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_jsonl_chunks(self, filepath: str, chunk_size: int = 10000) -> Generator[List[Dict], None, None]:
        """
        Load JSONL file in chunks to manage memory.
        Yields chunks of documents.
        """
        chunk = []
        total_loaded = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        chunk.append(doc)
                        total_loaded += 1
                        
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
            
            # Yield remaining documents
            if chunk:
                yield chunk
        
        print(f"✅ Total loaded: {total_loaded} documents")
    
    def get_total_lines(self, filepath: str) -> int:
        """Count total lines in JSONL file without loading into memory"""
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count