import torch
import torch.nn.functional as F
import re
from model import get_medimicro_300m_model
from tokenizer import get_tokenizer

def generate_safe_response(model, tokenizer, prompt, max_new_tokens=50, threshold=0.6):
    device = next(model.parameters()).device
    
    # 1. Format input with ChatML template
    messages = [
        {"role": "system", "content": "You are a helpful medical AI assistant. Greet the user politely if they say hello."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
    
    model.eval()
    generated_tokens = []
    token_probs = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            if hasattr(logits, "logits"):
                logits = logits.logits
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

def check_safety_prompt(prompt):
    dangerous_keywords = ["take 50mg", "drink bleach", "suicide", "cure cancer"]
    for keyword in dangerous_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', prompt, re.IGNORECASE):
            return False
    return True

def stream_safe_response(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_p=1.0, threshold=0.6):
    if not check_safety_prompt(prompt):
        yield "I cannot provide specific medical advice or respond to this prompt safely. Please contact emergency services or a medical professional."
        return

    device = next(model.parameters()).device
    
    # 1. Format input with ChatML template
    messages = [
        {"role": "system", "content": "You are a helpful medical AI assistant. Greet the user politely if they say hello."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
    
    model.eval()
    token_probs = []
    generated_tokens = []
    current_text = ""
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            if hasattr(logits, "logits"):
                logits = logits.logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            
            if temperature == 0:
                next_token = torch.argmax(probs)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze()
                
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)
                
            next_token_prob = probs[next_token[0]].item()
            
            if next_token[0] == tokenizer.eos_token_id:
                break
                
            token_probs.append(next_token_prob)
            generated_tokens.append(next_token[0].item())
            
            # 2. Uncertainty Detection on the fly
            if len(token_probs) >= 5:
                avg_prob = sum(token_probs) / len(token_probs)
                if avg_prob < threshold:
                    yield "\n\n[SAFETY TRIGGER]: I am not certain about this. Please consult a doctor."
                    break

            full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            new_text = full_text[len(current_text):]
            current_text = full_text
            yield new_text
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Loading model for inference...")
    
    # Load from hf_export
    model = AutoModelForCausalLM.from_pretrained("hf_export").to(device)
    tokenizer = AutoTokenizer.from_pretrained("hf_export")
    
    prompt = "What is hypertension?"
    print(f"\nUser: {prompt}")
    response = generate_safe_response(model, tokenizer, prompt)
    print(f"Medi-Micro: {response}")
