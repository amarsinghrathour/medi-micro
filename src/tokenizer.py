from transformers import AutoTokenizer

def get_tokenizer():
    """
    Loads an off-the-shelf tokenizer from HuggingFace.
    We use TinyLlama as it has the standard 32,000 vocab size 
    expected by our custom model architecture.
    """
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Ensure it has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

if __name__ == "__main__":
    # Test tokenizer
    tok = get_tokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    
    # Test chat template
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False)
    print("Chat Template output:")
    print(prompt)
