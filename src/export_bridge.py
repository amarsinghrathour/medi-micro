import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from model import get_medimicro_300m_model

def export_to_hf(custom_checkpoint_path="checkpoints/medi_micro_custom.pt", output_dir="hf_export"):
    print(f"Loading custom model from {custom_checkpoint_path}...")
    
    # 1. Initialize our custom model and load weights
    custom_model = get_medimicro_300m_model()
    custom_model.load_state_dict(torch.load(custom_checkpoint_path, map_location="cpu"))
    
    # 2. Define standard HF LlamaConfig that perfectly matches our custom architecture math
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=2816,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
    )
    
    # 3. Initialize HF Llama model
    print("Initializing HuggingFace Llama model...")
    hf_model = LlamaForCausalLM(config)
    hf_state_dict = hf_model.state_dict()
    
    custom_state_dict = custom_model.state_dict()
    
    # 4. The Bridge: Mapping Custom Keys to HF Keys
    print("Bridging weights...")
    
    # Embeddings
    hf_state_dict["model.embed_tokens.weight"] = custom_state_dict["tok_embeddings.weight"]
    
    # Output
    hf_state_dict["model.norm.weight"] = custom_state_dict["norm.weight"]
    hf_state_dict["lm_head.weight"] = custom_state_dict["output.weight"]
    
    # Layers
    for i in range(config.num_hidden_layers):
        prefix_custom = f"layers.{i}"
        prefix_hf = f"model.layers.{i}"
        
        # Attention Projections
        hf_state_dict[f"{prefix_hf}.self_attn.q_proj.weight"] = custom_state_dict[f"{prefix_custom}.attention.wq.weight"]
        hf_state_dict[f"{prefix_hf}.self_attn.k_proj.weight"] = custom_state_dict[f"{prefix_custom}.attention.wk.weight"]
        hf_state_dict[f"{prefix_hf}.self_attn.v_proj.weight"] = custom_state_dict[f"{prefix_custom}.attention.wv.weight"]
        hf_state_dict[f"{prefix_hf}.self_attn.o_proj.weight"] = custom_state_dict[f"{prefix_custom}.attention.wo.weight"]
        
        # MLPs
        hf_state_dict[f"{prefix_hf}.mlp.gate_proj.weight"] = custom_state_dict[f"{prefix_custom}.feed_forward.w1.weight"]
        hf_state_dict[f"{prefix_hf}.mlp.down_proj.weight"] = custom_state_dict[f"{prefix_custom}.feed_forward.w2.weight"]
        hf_state_dict[f"{prefix_hf}.mlp.up_proj.weight"] = custom_state_dict[f"{prefix_custom}.feed_forward.w3.weight"]
        
        # Norms
        hf_state_dict[f"{prefix_hf}.input_layernorm.weight"] = custom_state_dict[f"{prefix_custom}.attention_norm.weight"]
        hf_state_dict[f"{prefix_hf}.post_attention_layernorm.weight"] = custom_state_dict[f"{prefix_custom}.ffn_norm.weight"]
    
    # 5. Load bridged weights into HF model
    hf_model.load_state_dict(hf_state_dict, strict=True)
    
    # 6. Save in standard HF format (config.json, safetensors/pytorch_model.bin)
    print(f"Exporting to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)
    
    # 7. Export the tokenizer as well
    print("Exporting tokenizer...")
    from tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    tokenizer.save_pretrained(output_dir)
    print("✅ Export complete! Model is ready for Ollama / llama.cpp / Transformers.js")

if __name__ == "__main__":
    export_to_hf()
