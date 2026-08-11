# Medi-Micro Engineering Document

## Architecture Overview

Medi-Micro is designed as a highly efficient, decoder-only Transformer specifically targeted for Apple Silicon (M4) using PyTorch's Metal Performance Shaders (MPS) backend. 

Crucially, **the architecture is built completely from scratch in PyTorch**, but its mathematical formulation aligns perfectly with the standard Llama architecture. This allows us to train a custom model from the ground up, while retaining the ability to export it to popular inference systems (like Ollama and Transformers.js).

### Model Parameters (~266M)
To achieve a model size under 1 GB (specifically ~550 MB in FP16), the architecture is configured as follows:
- **Vocabulary Size (`vocab_size`)**: 32,000 (Uses Hugging Face `TinyLlama` tokenizer for compatibility)
- **Hidden Dimension (`d_model`)**: 1024
- **Number of Layers (`num_layers`)**: 16
- **Number of Attention Heads (`num_heads`)**: 16
- **Feed-Forward Dimension (`hidden_dim`)**: 2816
- **Max Sequence Length**: 1024

#### Parameter Breakdown:
- **Embeddings**: 32,000 × 1024 ≈ 32.7M
- **Attention**: 16 × (4 × 1024²) ≈ 67.1M
- **Feed-Forward**: 16 × (2 × 1024 × 2816 + 2816 × 1024) ≈ 138.4M
- **Output Projection**: 32,000 × 1024 ≈ 32.7M
- **Total**: ~270M parameters.

### Tokenization Strategy
To avoid reinventing the wheel and ensure ecosystem compatibility, we use the `TinyLlama` tokenizer from Hugging Face. This provides a robust 32,000 subword vocabulary out-of-the-box, complete with standard special tokens required for ChatML formatting.

### Advanced Architectural Choices (Implemented From Scratch)
- **Rotary Positional Embeddings (RoPE)**: Provides better relative position information without the parameter overhead of learned absolute embeddings.
- **RMSNorm**: Faster and computationally lighter than standard LayerNorm.
- **SwiGLU**: Modern activation function used in the Feed-Forward networks.
- **Mixed Precision**: We use PyTorch `autocast` tailored for MPS to speed up training while maintaining numerical stability.

## The Export Bridge

Because we wrote the model completely from scratch using our own PyTorch classes (`MediMicroModel`, `TransformerBlock`), systems like Ollama cannot natively read our checkpoints.

To solve this, we implemented an **Export Bridge** (`src/export_bridge.py`). 
This script:
1. Instantiates our custom PyTorch model and loads the trained weights.
2. Initializes a standard Hugging Face `LlamaForCausalLM` with identical dimensions.
3. Dynamically maps our custom state dictionary keys (e.g., `layers.0.attention.wq.weight`) to the standard Hugging Face keys (`model.layers.0.self_attn.q_proj.weight`).
4. Saves the model using `save_pretrained()`, generating the standard `config.json` and `safetensors` files that Ollama and Transformers.js require.

## Safety & Hallucination Mitigation Protocol

A ~300M parameter model has a limited capacity for world knowledge. Therefore, the core safety mechanism is teaching the model to *refuse* to answer when uncertain or when the query is out of domain.

### 1. Data-Level Strategy (Synthetic Refusals)
Approximately 20% of the training dataset consists of synthetic "Negative Examples."
- **Out of Scope**: E.g., "Who won the World Cup?" -> "My knowledge is focused on health topics. I cannot answer non-medical questions."
- **Too Broad / Dangerous**: E.g., "What is the cure for cancer?" -> "I cannot provide specific cures for complex diseases. Please consult an oncologist."

### 2. Inference-Level Uncertainty Detection
During generation (`src/inference.py`), we monitor the model's confidence logits.
If the average probability of the generated tokens falls below a defined threshold (e.g., `< 0.6`), the generation is halted, and a fallback refusal message is triggered: *"I am not certain about this. Please consult a doctor."*

### 3. Keyword Post-processing
A lightweight regex filter scans the generated output for high-risk terms (e.g., "take 50mg", "drink bleach", "suicide"). If detected, the response is overridden with a safe, pre-scripted message.
