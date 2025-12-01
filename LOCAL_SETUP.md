# Local Model Setup Guide

Complete guide for running evaluations with local models on your GPU.

## Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **VRAM Requirements**:
  - 3B model: ~6GB (4-bit) or ~12GB (FP16)
  - 7-8B model: ~8GB (4-bit) or ~28GB (FP16)
  - 13B model: ~12GB (4-bit) or ~52GB (FP16)

## 1. Install Dependencies

```bash
cd /Users/faiqhilman/Projects/llm-eval-harness

# For local models (includes torch, transformers)
pip install -e .[local]

# For finetuning (includes all training deps)
pip install -e .[finetune]

# For analysis (includes pandas, matplotlib, jupyter)
pip install -e .[analysis]

# Or install everything at once
pip install -e .[all]
```

**Note**: See [INSTALL.md](INSTALL.md) for detailed installation options.

## 2. Test Your Setup

```bash
# Verify CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run pipeline test (no GPU required)
python3 scripts/test_pipeline.py
```

## 3. Recommended Models to Try

### Small Models (Good for Testing)
```bash
# Llama 3.2 3B (fastest, ~6GB VRAM in 4-bit)
--model meta-llama/Llama-3.2-3B --model-type local --load-in-4bit

# Phi-3 Mini (very efficient, ~4GB VRAM in 4-bit)
--model microsoft/Phi-3-mini-4k-instruct --model-type local --load-in-4bit

# Gemma 2B (Google, ~3GB VRAM in 4-bit)
--model google/gemma-2b-it --model-type local --load-in-4bit
```

### Medium Models (Better Quality)
```bash
# Llama 3.2 8B (~8GB VRAM in 4-bit)
--model meta-llama/Llama-3.1-8B-Instruct --model-type local --load-in-4bit

# Mistral 7B (very good, ~8GB VRAM in 4-bit)
--model mistralai/Mistral-7B-Instruct-v0.3 --model-type local --load-in-4bit
```

## 4. Run Local Model Evaluation

### JSON Extraction Task

```bash
# Small batch test (5 samples)
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --limit 5

# Full test set (120 samples)
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit
```

**First run**: Model downloads from HuggingFace (~6GB), takes 5-10 minutes
**Subsequent runs**: Instant loading from cache

### Q&A Task

```bash
# Q&A with API judge (recommended)
export OPENAI_API_KEY="sk-..."

python3 scripts/run_eval.py \
  --task qa \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --judge-model gpt-4o-mini \
  --limit 10
```

## 5. Compare Local vs API Models

```bash
# Step 1: Run API baseline
export OPENAI_API_KEY="sk-..."
python3 scripts/run_eval.py \
  --task json \
  --model gpt-4o-mini \
  --limit 20

# Note the run_id from output, e.g., run_20251201_120000_abc123

# Step 2: Run local model
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --limit 20

# Note the run_id, e.g., run_20251201_130000_def456

# Step 3: Compare
python3 scripts/compare_runs.py \
  --baseline run_20251201_120000_abc123 \
  --current run_20251201_130000_def456 \
  --output comparison.md

# View comparison
cat comparison.md
```

## 6. Train QLoRA Adapter

```bash
# Train adapter on JSON extraction task
python3 scripts/train_qlora.py

# This will:
# - Download Llama-3.2-3B (~6GB)
# - Train LoRA adapter for 2 epochs (~30 mins on RTX 4090)
# - Save adapter to data/lora_adapters/json_v1/
# - Use ~18GB VRAM (4-bit model + training overhead)
```

Training output:
```
QLoRA Training - JSON Extraction
Base model: meta-llama/Llama-3.2-3B
Output: data/lora_adapters/json_v1
Epochs: 2
LoRA r=16, alpha=32
Effective batch size: 16
...
Training complete!
✅ Adapter saved to: data/lora_adapters/json_v1
```

## 7. Evaluate Finetuned Model

After training, you can load the adapter:

```python
# Create a script to load model + adapter
# (vLLM implementation needed for production use)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(
    model,
    "data/lora_adapters/json_v1"
)

# Now use model for evaluation
```

## 8. Memory Optimization Tips

### If you run out of VRAM:

**Use 4-bit quantization** (default in this setup):
```bash
--load-in-4bit
```

**Reduce batch size** in training:
```python
# In src/llm_eval/finetune/qlora_trainer.py
per_device_train_batch_size: int = 1  # Already set
gradient_accumulation_steps: int = 16  # Effective batch = 16
```

**Reduce sequence length**:
```python
max_seq_length: int = 256  # Down from 512
```

**Use smaller model**:
```bash
# Try Phi-3 Mini instead of Llama
--model microsoft/Phi-3-mini-4k-instruct
```

## 9. Typical Workflow

```bash
# 1. Quick test with small model
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --limit 5

# 2. Run API baseline for comparison
export OPENAI_API_KEY="sk-..."
python3 scripts/run_eval.py \
  --task json \
  --model gpt-4o-mini \
  --limit 20

# 3. Run full local evaluation
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --limit 20

# 4. Compare results
python3 scripts/compare_runs.py \
  --baseline <api_run_id> \
  --current <local_run_id>

# 5. Train LoRA adapter
python3 scripts/train_qlora.py

# 6. Evaluate finetuned model
# (Need to implement adapter loading - see section 7)

# 7. Analyze in Jupyter
jupyter notebook notebooks/results_analysis.ipynb
```

## 10. Troubleshooting

### "CUDA out of memory"
- Add `--load-in-4bit` flag
- Reduce `--limit` to fewer samples
- Try smaller model (Phi-3 Mini, Gemma 2B)
- Close other GPU applications

### "Model not found on HuggingFace"
- Check model name spelling
- May need HuggingFace token for gated models:
  ```bash
  # Login to HuggingFace
  pip install huggingface-hub
  huggingface-cli login
  ```

### "Slow generation"
- Normal for first run (model download)
- 4-bit models are slower than FP16 but use less memory
- Consider using vLLM for production (faster inference)

### "Import errors"
```bash
# Reinstall dependencies
pip install --upgrade torch transformers accelerate bitsandbytes
```

## 11. Expected Performance

### JSON Extraction (120 samples)

| Model | Time | Accuracy (est) | VRAM |
|-------|------|---------------|------|
| GPT-4o-mini (API) | ~2 min | 85-90% | 0GB |
| Llama-3.2-3B (4-bit) | ~10 min | 60-70% | 6GB |
| Llama-3.2-3B + LoRA | ~10 min | 75-85% | 6GB |
| Mistral-7B (4-bit) | ~15 min | 75-80% | 8GB |

### Training Times (RTX 4090, 2000 samples, 2 epochs)

| Model | Time | VRAM | Adapter Size |
|-------|------|------|--------------|
| Llama-3.2-3B | ~30 min | ~18GB | ~50MB |
| Llama-3.1-8B | ~1 hour | ~22GB | ~75MB |

## 12. Model Registry (configs/models.yaml)

Update this to add your models:

```yaml
models:
  # API models
  gpt-4o-mini:
    type: api
    provider: openai

  # Local models
  llama-3.2-3b:
    type: local
    model_path: meta-llama/Llama-3.2-3B
    load_in_4bit: true

  llama-3.2-3b-json-lora:
    type: local
    model_path: meta-llama/Llama-3.2-3B
    adapter_path: data/lora_adapters/json_v1
    load_in_4bit: true
```

## 13. Next Steps

1. **Benchmark**: Run evaluations on 3-5 models
2. **Compare**: Use `compare_runs.py` to see differences
3. **Finetune**: Train LoRA on best base model
4. **Re-evaluate**: Test finetuned model vs baseline
5. **Analyze**: Use Jupyter notebook for deep dive
6. **Production**: Implement vLLM serving for fast inference

## Questions?

Check:
- [README.md](README.md) - Main documentation
- [COMPLETE.md](COMPLETE.md) - Full feature list
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
