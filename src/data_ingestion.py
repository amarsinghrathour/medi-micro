import os
import json
import re
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Simple synthetic refusal examples
SYNTHETIC_REFUSALS = [
    {"messages": [{"role": "user", "content": "What is the cure for cancer?"}, {"role": "assistant", "content": "I am an AI model trained on general health information. I cannot provide a specific cure for cancer. Please consult a medical professional."}]},
    {"messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}, {"role": "assistant", "content": "My knowledge is focused on health topics. I cannot answer non-medical questions."}]},
    {"messages": [{"role": "user", "content": "What is the best stock to buy right now?"}, {"role": "assistant", "content": "I am a medical AI and cannot provide financial advice."}]},
    {"messages": [{"role": "user", "content": "How much ibuprofen should I take for a headache?"}, {"role": "assistant", "content": "I cannot provide specific medical dosage advice. Please follow the instructions on the package or consult a pharmacist."}]}
]

# Simple synthetic greeting examples
SYNTHETIC_GREETINGS = [
    {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! I am Medi-Micro, your medical AI assistant. How can I help you today?"}]},
    {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi there! I'm here to assist you with medical questions. What would you like to know?"}]},
    {"messages": [{"role": "user", "content": "hey"}, {"role": "assistant", "content": "Hello! How can I assist you with your health or medical questions?"}]},
    {"messages": [{"role": "user", "content": "how are you?"}, {"role": "assistant", "content": "I'm just a computer program, but I'm ready to help answer your medical questions!"}]}
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove extremely short text
    if len(text) < 20:
        return ""
    # Simple regex to filter out potential phone numbers/emails (PII)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REMOVED]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REMOVED]', text)
    return text.strip()

def process_medqa():
    print("Downloading MedQA dataset from HuggingFace...")
    # Using a common MedQA instruction dataset
    dataset = load_dataset("medalpaca/medical_meadow_medqa", token=HF_TOKEN, split="train")
    
    processed = []
    print("Filtering and formatting MedQA...")
    for item in tqdm(dataset):
        if not isinstance(item, dict):
            continue
            
        instruction = clean_text(item.get("instruction", ""))
        inp = clean_text(item.get("input", ""))
        output = clean_text(item.get("output", ""))
        
        if len(instruction) > 20 and len(output) > 20:
            user_msg = f"{instruction}\n{inp}".strip()
            processed.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": output}
                ]
            })
    return processed

def process_pubmed():
    print("Downloading PubMed dataset from HuggingFace...")
    # Using the medalpaca PubMed causal dataset
    dataset = load_dataset("medalpaca/medical_meadow_pubmed_causal", token=HF_TOKEN, split="train")
    
    processed = []
    print("Filtering and formatting PubMed...")
    for item in tqdm(dataset):
        if not isinstance(item, dict):
            continue
            
        instruction = clean_text(item.get("instruction", ""))
        inp = clean_text(item.get("input", ""))
        output = clean_text(item.get("output", ""))
        
        if len(instruction) > 20 and len(output) > 20:
            user_msg = f"{instruction}\n{inp}".strip()
            processed.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": output}
                ]
            })
    return processed

def main():
    os.makedirs("data", exist_ok=True)
    out_file = "data/train.jsonl"
    
    medqa_data = process_medqa()
    pubmed_data = process_pubmed()
    
    final_data = medqa_data + pubmed_data + (SYNTHETIC_REFUSALS * 100) + (SYNTHETIC_GREETINGS * 100) # Duplicate refusals/greetings to give them weight
    
    print(f"Writing {len(final_data)} samples to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item) + "\n")
            
    print("Data ingestion complete.")

if __name__ == "__main__":
    main()
