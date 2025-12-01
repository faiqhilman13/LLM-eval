# Installation Guide

## Quick Install

### Option 1: API Models Only (Minimal)
```bash
# Just OpenAI API support
pip install -e .
```

### Option 2: Local Models (HuggingFace)
```bash
# Includes torch + transformers (~4GB download)
pip install -e .[local]
```

### Option 3: Finetuning (QLoRA)
```bash
# Includes everything for training (~6GB download)
pip install -e .[finetune]
```

### Option 4: Everything
```bash
# All features (~10GB download)
pip install -e .[all]
```

## Installation Extras Explained

| Extra | Use Case | Includes | Size |
|-------|----------|----------|------|
| *none* | API models only | openai, pyyaml | ~50MB |
| `[local]` | Run local models | torch, transformers | ~4GB |
| `[finetune]` | Train LoRA adapters | + peft, bitsandbytes, trl | ~6GB |
| `[serve]` | vLLM serving | vllm | ~3GB |
| `[analysis]` | Jupyter notebooks | pandas, matplotlib, jupyter | ~500MB |
| `[dev]` | Development | pytest, black, ruff | ~100MB |
| `[all]` | Everything | All above | ~10GB |

## Recommended Setups

### For API Benchmarking Only
```bash
pip install -e .
```
**Use case**: Compare GPT-4o-mini, Claude, etc.

### For Local Model Evaluation
```bash
pip install -e .[local,analysis]
```
**Use case**: Evaluate Llama, Mistral, etc. + analyze results

### For Full Local Development
```bash
pip install -e .[finetune,analysis,dev]
```
**Use case**: Train LoRA, evaluate, and develop

### For Production Deployment
```bash
pip install -e .[all]
```
**Use case**: Full harness with all features

## Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/faiqhilman13/LLM-eval.git
cd LLM-eval
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Based on Your Needs

**API only:**
```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 5
```

**Local models:**
```bash
pip install -e .[local]
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --limit 5
```

**Finetuning:**
```bash
pip install -e .[finetune]
python3 scripts/train_qlora.py
```

## Troubleshooting

### "No module named 'torch'"
```bash
# Install local extra
pip install -e .[local]
```

### "No module named 'peft'"
```bash
# Install finetune extra
pip install -e .[finetune]
```

### "No module named 'pandas'"
```bash
# Install analysis extra
pip install -e .[analysis]
```

### CUDA Issues
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, you may need to reinstall torch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Disk Space
- Minimal install: ~50MB
- With local models: ~4GB
- With finetuning: ~6GB
- Full install: ~10GB

Make sure you have enough space before installing `[all]`.

## Updating

```bash
# Pull latest changes
git pull

# Reinstall (keeps your extras)
pip install -e .[local]  # Or whatever extras you're using
```

## Uninstall

```bash
pip uninstall llm-eval-harness
```

## GPU Requirements

| Feature | VRAM Needed |
|---------|-------------|
| API models | 0GB (no GPU) |
| Llama-3.2-3B (4-bit) | 6GB |
| Llama-3.1-8B (4-bit) | 8GB |
| QLoRA training (3B) | 16GB |
| QLoRA training (8B) | 22GB |

## Quick Test

After installation, verify it works:

```bash
# Test without GPU (mock test)
python3 scripts/test_pipeline.py

# Test with API
export OPENAI_API_KEY="sk-..."
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 2
```

## Next Steps

See:
- [START_HERE.md](START_HERE.md) - Quick commands
- [LOCAL_SETUP.md](LOCAL_SETUP.md) - Local model guide
- [README.md](README.md) - Full documentation
