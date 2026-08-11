import torch
import torch.nn.functional as F
import re
from model import get_medimicro_300m_model
from tokenizer import get_tokenizer

def generate_safe_response(model, tokenizer, prompt, max_new_tokens=50, threshold=0.6):
    device = next(model.parameters()).device
    
    # 1. Format input with ChatML template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
    
    model.eval()
    generated_tokens = []
    token_probs = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Simple greedy decoding for now (can add temperature later)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs)
            next_token_prob = probs[next_token].item()
            
            if next_token == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            token_probs.append(next_token_prob)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            
    # 2. Uncertainty Detection
    if len(token_probs) > 0:
        avg_prob = sum(token_probs) / len(token_probs)
        if avg_prob < threshold:
            print(f"[SAFETY TRIGGER] High uncertainty detected (avg_prob: {avg_prob:.2f} < {threshold})")
            return "I am not certain about this. Please consult a doctor."
            
    # 3. Decode output
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 4. Keyword Filtering Post-processing
    dangerous_keywords = ["take 50mg", "drink bleach", "suicide", "cure cancer"]
    for keyword in dangerous_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', output_text, re.IGNORECASE):
            print(f"[SAFETY TRIGGER] Dangerous keyword detected: '{keyword}'")
            return "I cannot provide specific medical advice or dosage information. Please contact emergency services or a medical professional."
            
    return output_text.strip()

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Loading model for inference...")
    
    model = get_medimicro_300m_model().to(device)
    # Normally we would load weights here:
    # model.load_state_dict(torch.load("checkpoints/medi_micro_custom.pt"))
    
    tokenizer = get_tokenizer()
    
    prompt = "What is hypertension?"
    print(f"\nUser: {prompt}")
    response = generate_safe_response(model, tokenizer, prompt)
    print(f"Medi-Micro: {response}")
