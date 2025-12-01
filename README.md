# LLM Evaluation & Finetuning Harness

Production-grade evaluation framework for comparing base models, API models, and finetuned variants. Demonstrates end-to-end ML rigor: data curation → QLoRA finetuning → vLLM serving → observability → CI regression gates.

## 🎯 Key Features

- **Multi-model evaluation**: Compare API models (GPT-4o-mini), local models (Llama 3), and finetuned variants
- **Dual scoring**: Deterministic validators + LLM-as-judge
- **QLoRA finetuning**: Memory-efficient training on consumer GPUs (RTX 3090/4090)
- **vLLM serving**: High-performance inference with LoRA adapter support
- **SQLite metrics**: Lightweight storage with CSV/Prometheus export
- **CI/CD regression gates**: Automated quality/latency/cost monitoring
- **OpenTelemetry traces**: Optional distributed tracing (disabled by default)

## ✅ Implementation Status

**M0-M4 COMPLETE** - Full production-ready harness!

- ✅ M1: Model adapters (API + HuggingFace) + JSON task + deterministic scoring
- ✅ M2: LLM-as-judge scorer + Q&A task + threshold system + comparison CLI
- ✅ M3: CSV export + analysis notebooks + optional OpenTelemetry tracing
- ✅ M4: QLoRA trainer + HuggingFace adapter + CI/CD pipelines

**Datasets**: 4,300 samples (2,150 JSON + 2,150 Q&A) across train/val/test splits

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16-24GB VRAM (for local finetuning/serving)
- OpenAI API key (for baselines)
- HuggingFace token (for model downloads)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-eval-harness.git
cd llm-eval-harness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run Your First Evaluation

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run JSON extraction task (120 test samples)
python3 scripts/run_eval.py --task json --model gpt-4o-mini

# Run Q&A task with LLM judge
python3 scripts/run_eval.py --task qa --model gpt-4o-mini --judge-model gpt-4o-mini

# Test with small batch
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 5

# Compare two runs
python3 scripts/compare_runs.py \
  --baseline <run_id_1> \
  --current <run_id_2> \
  --fail-on-regression
```

## 📁 Project Structure

```
llm-eval-harness/
├── src/llm_eval/          # Core package
│   ├── models/            # Model adapters (API, local, vLLM)
│   ├── tasks/             # Task definitions (JSON, Q&A)
│   ├── scorers/           # Scoring (deterministic, LLM-judge)
│   ├── metrics/           # Metrics collection & storage
│   └── finetune/          # QLoRA training
├── data/
│   ├── tasks/             # Test datasets
│   ├── lora_adapters/     # Finetuned adapters
│   └── runs/              # Evaluation results
├── configs/               # Model/task configurations
├── scripts/               # CLI tools
├── notebooks/             # Analysis notebooks
└── tests/                 # Test suite
```

## 🔧 Usage

### 1. Running Evaluations

```bash
# List available models
python scripts/run_eval.py --list-models

# Run evaluation
python scripts/run_eval.py \
  --task json_extraction \
  --model llama3-8b-base \
  --output data/runs/my-experiment

# Run with custom config
python scripts/run_eval.py --config configs/evaluations/baseline.yaml
```

### 2. Finetuning with QLoRA

```bash
# Train LoRA adapter
python scripts/train_qlora.py \
  --task json_extraction \
  --base-model meta-llama/Llama-3-8B \
  --output data/lora_adapters/json_v1

# Evaluation finetuned model
python scripts/run_eval.py \
  --task json_extraction \
  --model llama3-8b-json-lora
```

### 3. Serving with vLLM

```bash
# Start vLLM server
python scripts/serve_vllm.py \
  --model meta-llama/Llama-3-8B \
  --adapters data/lora_adapters/json_v1

# Server runs on http://localhost:8001
```

### 4. Analyzing Results

```bash
# Export metrics to CSV
python scripts/export_metrics.py \
  --format csv \
  --output results.csv

# Open analysis notebook
jupyter notebook notebooks/results_analysis.ipynb
```

## 🎓 Tasks

### JSON Extraction
- **Goal**: Extract structured data from natural language
- **Test set**: 150 cases (easy, medium, hard, domain-specific)
- **Training set**: 2000 samples
- **Metrics**: Schema validity, field accuracy, type correctness

### Q&A (Question Answering)
- **Goal**: Answer questions given context
- **Test set**: 150 cases (factual, reasoning, multi-hop)
- **Training set**: 2000 samples
- **Metrics**: LLM-judge score (1-5), faithfulness, completeness

## ⚙️ Configuration

### Model Registry (`configs/models.yaml`)

```yaml
models:
  gpt-4o-mini:
    type: api
    provider: openai
    cost_per_1k_prompt: 0.00015

  llama3-8b-base:
    type: local
    model_path: meta-llama/Llama-3-8B

  llama3-8b-json-lora:
    type: vllm_lora
    base_model: meta-llama/Llama-3-8B
    lora_path: data/lora_adapters/json_v1
    gpu_memory_utilization: 0.85
```

### Evaluation Config (`configs/evaluations/baseline.yaml`)

```yaml
task: json_extraction
models:
  - gpt-4o-mini
  - llama3-8b-base
output_dir: data/runs/baseline
enable_tracing: false
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/llm_eval --cov-report=html

# Run specific test category
pytest -m integration
pytest -m "not slow"
```

## 📊 Observability

### SQLite Metrics (Always Enabled)
- Metrics stored in `data/metrics.db`
- Export to CSV/Prometheus
- Query via Python/SQL

### OpenTelemetry (Optional)
```bash
# Enable tracing
export ENABLE_OTEL=true
export OTEL_ENDPOINT=http://localhost:4317

# Run evaluation with tracing
python scripts/run_eval.py --task json --model llama3-8b-base
```

## 🚦 CI/CD

GitHub Actions workflows:
- **Unit tests**: Run on every PR
- **Regression tests**: Run on main branch (weekly)
- **Quality gates**: Fail if accuracy/latency/cost thresholds exceeded

## 💰 Cost Optimization

**Local-first development**:
- All training on your GPU: $0
- vLLM serving: $0
- API baselines only: ~$10-20/month

**Tips**:
- Use API models sparingly (baselines only)
- Batch evaluations to minimize API calls
- Test on smaller models first (TinyLlama)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `black` and `ruff` for formatting
5. Submit a pull request

## 📚 Documentation

- [Architecture](docs/architecture.md)
- [Adding Tasks](docs/adding_tasks.md)
- [Finetuning Guide](docs/finetuning_guide.md)
- [Quickstart](docs/quickstart.md)

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

Built with:
- [vLLM](https://github.com/vllm-project/vllm) - Fast inference
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient finetuning
- [Transformers](https://github.com/huggingface/transformers) - Model library
- [OpenTelemetry](https://opentelemetry.io/) - Observability

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/llm-eval-harness](https://github.com/yourusername/llm-eval-harness)
