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
2. **Train Model**: Runs the custom from-scratch PyTorch training loop utilizing MPS mixed precision.
   ```bash
   uv run python src/train.py
   ```
3. **Export Bridge**: Converts your custom PyTorch checkpoint into the standard Hugging Face format for Ollama / Transformers.js compatibility.
   ```bash
   uv run python src/export_bridge.py
   ```
4. **Safe Inference**: Test the model with uncertainty thresholding active.
   ```bash
   uv run python src/inference.py
   ```

## Development Phases
Refer to `docs/engineering.md` for a deeper dive into the model architecture, the math behind the parameter count, and the safety protocols.
