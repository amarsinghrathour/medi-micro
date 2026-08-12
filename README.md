# Medi-Micro

Medi-Micro is a ~300M parameter decoder-only Transformer tailored for medical Q&A on Apple Silicon (M4). The primary goal is achieving high accuracy on known medical topics and zero hallucinations on unknown topics through an uncertainty detection mechanism.

The model is built **entirely from scratch in PyTorch**, but utilizes an Export Bridge to align perfectly with the standard Llama architecture, allowing you to run the final model natively on cross-platform inference engines like **Ollama** and **Transformers.js**.

## Hardware Requirements
- Apple Silicon (M4 recommended, M1/M2/M3 supported)
- macOS 13+
- At least 8GB RAM (The ~550MB FP16 model easily fits in memory)

## Setup

1. **Install `uv`**:
   Ensure you have `uv` installed for fast Python dependency management.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and Install**:
   ```bash
   git clone <your_repo_url> medi-micro
   cd medi-micro
   uv sync
   ```
   *Note: If a team member adds a new dependency to `pyproject.toml`, or if you ever encounter a `ModuleNotFoundError`, run `uv sync` again to update your local environment.*

3. **Configure Environment Variables**:
   Copy the example environment file and add your HuggingFace read token.
   ```bash
   cp .env.example .env
   # Edit .env and set HF_TOKEN
   ```

4. **Verify Hardware**:
   Run the verification script to ensure PyTorch detects and utilizes the MPS backend correctly.
   ```bash
   uv run python check_hardware.py
   ```

## Workflow Pipeline

You can run the entire end-to-end pipeline (hardware check, data ingestion, training, and export) with a single command:
```bash
uv run python main.py
```

Alternatively, you can run individual steps:

1. **Data Ingestion**: Downloads the MedQA and PubMed Causal datasets, formats them into ChatML, safely handles malformed rows, and injects synthetic safety data.
   ```bash
   uv run python src/data_ingestion.py
   ```
2. **Fine-tune the Model**
Train the model using LoRA (Low-Rank Adaptation) on TinyLlama. This will download the base model, fine-tune it on your local data, and save the merged model to `hf_export`.
```bash
uv run python src/train.py
```

3. **Start the Chat App**
Once the model is exported to `hf_export`, you can launch the GUI!
```bash
uv run python src/chat_app.py
```

## Development Phases
Refer to `docs/engineering.md` for a deeper dive into the model architecture, the math behind the parameter count, and the safety protocols.
