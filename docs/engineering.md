# Medi-Micro Engineering Document

## Architecture Overview

Medi-Micro is designed as a highly efficient medical Q&A model targeting Apple Silicon (M4). Rather than training a model completely from scratch, we use **LoRA (Low-Rank Adaptation)** to fine-tune the pre-trained **TinyLlama-1.1B** base model.

This approach offers significant advantages:
- **Efficiency**: LoRA drastically reduces the number of trainable parameters by injecting small, low-rank matrices into the attention layers (`q_proj` and `v_proj`).
- **Performance**: HuggingFace's `Trainer` API is utilized for the training loop, which automatically handles optimal device placement (MPS on Apple Silicon), mixed-precision scaling, logging, and gradient accumulation.
- **Base Knowledge**: By leveraging TinyLlama (1.1B parameters), the model already possesses a strong foundational understanding of English grammar and general knowledge before we introduce specialized medical data.

### Tokenization Strategy
We use the standard `TinyLlama` tokenizer from Hugging Face. This provides a robust 32,000 subword vocabulary out-of-the-box, complete with standard special tokens required for ChatML formatting.

## Training & Export Pipeline

Because we use the HuggingFace `Trainer` and `peft` (Parameter-Efficient Fine-Tuning) libraries natively, the pipeline is vastly simplified:
1. **Train**: The `Trainer` optimizes the LoRA weights on our custom medical datasets.
2. **Merge & Export**: At the end of training, the adapter weights are seamlessly merged back into the base TinyLlama architecture (`merge_and_unload()`).
3. **Native Support**: The resulting merged model is saved using `save_pretrained()` to the `hf_export` directory. The output is a standard HuggingFace model (`config.json`, `safetensors`), entirely ready to be loaded by inference systems like Ollama or Transformers.js—**no manual export bridge required!**

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
