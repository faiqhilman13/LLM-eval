# Quickstart Guide

## M1 Complete - Ready to Use! ✅

The LLM Evaluation Harness M1 is fully functional. Here's how to get started.

## Install Dependencies

```bash
pip install openai pyyaml
```

## Set Your API Key

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

## Run Your First Evaluation

### Test with 2 samples
```bash
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 2
```

### Full test set (120 samples)
```bash
python3 scripts/run_eval.py --task json --model gpt-4o-mini
```

## Example Output

```
Initializing model: gpt-4o-mini
INFO:llm_eval.runner:Starting evaluation: json
INFO:llm_eval.runner:Model: gpt-4o-mini
INFO:llm_eval.runner:Samples: 120
INFO:llm_eval.runner:Run ID: run_20251201_180000_abc123de

INFO:llm_eval.runner:Processing sample 1/120: json_115
INFO:llm_eval.runner:Processing sample 2/120: json_142
...

Evaluation Complete!
Run ID: run_20251201_180000_abc123de
Total Samples: 120
Accuracy: 85.00%
Parse Rate: 98.33%

============================================================
EVALUATION RESULTS
============================================================
Run ID: run_20251201_180000_abc123de
Model: gpt-4o-mini
Task: json
Samples: 120

Metrics:
  total_samples: 120
  avg_exact_match: 0.8500
  avg_parse_success: 0.9833
  accuracy: 0.8500
  parse_rate: 0.9833
============================================================
```

## Results Storage

All results saved to: `data/metrics/eval_results.db`

Query your results:
```python
from src.llm_eval.metrics.storage import MetricsStorage

storage = MetricsStorage()
runs = storage.list_runs(limit=5)
for run in runs:
    print(f"{run['run_id']}: {run['model_name']} - {run['aggregate_metrics']['accuracy']:.2%}")
```

## Test Without API Key

Run mock tests to verify installation:
```bash
python3 scripts/test_pipeline.py
```

Should see:
```
✓ All tests completed successfully!
```

## CLI Options

```bash
python3 scripts/run_eval.py \
  --task json \              # Task type (json, qa, code)
  --model gpt-4o-mini \      # Model name
  --limit 10 \               # Number of samples (optional)
  --temperature 0.1 \        # Temperature (default: 0.0)
  --max-tokens 1000          # Max tokens (optional)
```

## What You're Evaluating

**JSON Extraction Task**: Extract structured data from text

Example input:
```
Email: eve@example.com, Phone: +1-555-8432, Active: True
```

Expected output:
```json
{
  "email": "eve@example.com",
  "phone": "+1-555-8432",
  "active": true
}
```

Metrics:
- **Accuracy**: Exact match rate (1.0 = perfect)
- **Parse Rate**: Valid JSON extraction rate

## Datasets Available

- **JSON Extraction**: 2,150 samples (easy/medium/hard)
  - Train: 2,000
  - Validation: 30
  - Test: 120

## Troubleshooting

**Error: "OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY="sk-..."
```

**Error: "Module not found"**
```bash
pip install openai pyyaml
```

**Test without API**
```bash
python3 scripts/test_pipeline.py
```

## Next Steps

1. Run evaluation on full test set
2. Try different models (gpt-4o, gpt-3.5-turbo)
3. Analyze results in SQLite database
4. Build QA and Code tasks (M2)

## Support

- Docs: `M1_COMPLETE.md`
- Session notes: `sessions/session2.md`
- Issues: Check code at `src/llm_eval/`

---

**Status**: Production-ready M1 ✅  
**Next**: M2 task expansion
