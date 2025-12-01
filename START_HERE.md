# 🚀 START HERE - Quick Commands

## Option 1: Test Without GPU (Mock Test)

```bash
# Verify everything works
python3 scripts/test_pipeline.py
```

## Option 2: Run with Local Model (GPU Required)

```bash
# Install dependencies first
pip install -e .[local]  # Includes torch + transformers

# Run small test (5 samples, ~2 minutes)
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --limit 5
```

**First run**: Downloads model (~6GB), takes 5-10 min
**Next runs**: Loads from cache, instant

## Option 3: Run with API Model (For Baseline)

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run evaluation (cheap, ~$0.05 for 20 samples)
python3 scripts/run_eval.py \
  --task json \
  --model gpt-4o-mini \
  --limit 20
```

## Compare Local vs API

```bash
# 1. Run both models
export OPENAI_API_KEY="sk-..."

# API model
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 10
# Note the run_id: run_20251201_120000_abc123

# Local model
python3 scripts/run_eval.py \
  --task json \
  --model meta-llama/Llama-3.2-3B \
  --model-type local \
  --load-in-4bit \
  --limit 10
# Note the run_id: run_20251201_130000_def456

# 2. Compare
python3 scripts/compare_runs.py \
  --baseline run_20251201_120000_abc123 \
  --current run_20251201_130000_def456
```

## Train Your First LoRA Adapter

```bash
# Requires: GPU with 16GB+ VRAM (RTX 3090/4090)
# Time: ~30 minutes for 2 epochs

python3 scripts/train_qlora.py
```

## Analyze Results

```bash
# Export to CSV
python3 -c "
from src.llm_eval.metrics.storage import MetricsStorage
from src.llm_eval.metrics.exporters import CSVExporter

storage = MetricsStorage()
exporter = CSVExporter(storage)
exporter.export_summary('results.csv', limit=10)
print('Exported to results.csv')
"

# Or use Jupyter
pip install jupyter pandas matplotlib seaborn
jupyter notebook notebooks/results_analysis.ipynb
```

## Recommended Models to Try

### Small & Fast (3-4B params)
- `meta-llama/Llama-3.2-3B` - ~6GB VRAM (4-bit)
- `microsoft/Phi-3-mini-4k-instruct` - ~4GB VRAM (4-bit)
- `google/gemma-2b-it` - ~3GB VRAM (4-bit)

### Better Quality (7-8B params)
- `meta-llama/Llama-3.1-8B-Instruct` - ~8GB VRAM (4-bit)
- `mistralai/Mistral-7B-Instruct-v0.3` - ~8GB VRAM (4-bit)

## Full Documentation

- [LOCAL_SETUP.md](LOCAL_SETUP.md) - Complete local setup guide
- [README.md](README.md) - Full documentation
- [COMPLETE.md](COMPLETE.md) - All features implemented
