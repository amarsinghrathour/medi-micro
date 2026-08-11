import os
from dotenv import load_dotenv

# Load .env before importing any Hugging Face libraries
load_dotenv()
if "HF_TOKEN" not in os.environ:
    print("⚠️ WARNING: HF_TOKEN not found in .env file. Hugging Face downloads may be rate limited.")

import check_hardware
from src.data_ingestion import main as ingest_data
from src.train import train as train_model
from src.export_bridge import export_to_hf

def main():
    print("=== Medi-Micro Pipeline ===")
    
    # 1. Check Hardware
    print("\n--- 1. Hardware Check ---")
    check_hardware.check_hardware()
    
    # 2. Data Ingestion
    print("\n--- 2. Data Ingestion ---")
    # ingest_data() will create data/train.jsonl if it doesn't exist.
    # Note: If it takes too long, you can comment this out after the first run.
    ingest_data()
    
    # 3. Train Model
    print("\n--- 3. Training Model ---")
    train_model()
    
    # 4. Export Model
    print("\n--- 4. Exporting Model ---")
    export_to_hf()
    
    print("\n✅ Pipeline complete! Your model is exported and ready.")

if __name__ == "__main__":
    main()
