# LLM Evaluation Harness - Session 2

**Date**: 2025-12-01  
**Status**: M1 COMPLETE ✅

## What Was Built

Built complete M1 pipeline from base to CLI in single session:

### Core Components (9/9 Complete)

1. **Model Interface** (`src/llm_eval/models/base.py`)
   - `BaseModel` abstract class
   - `Message` dataclass for chat format
   - `ModelResponse` with usage tracking
   - Support for single + batch generation

2. **OpenAI API Model** (`src/llm_eval/models/api_model.py`)
   - OpenAI SDK integration
   - Environment variable API key support
   - Usage token tracking
   - Batch generation (sequential for now)

3. **JSON Extraction Task** (`src/llm_eval/tasks/json_extraction.py`)
   - JSONL data loader (train/val/test splits)
   - `TaskSample` dataclass
   - Prompt formatter with schema injection
   - System prompt for extraction

4. **Deterministic Scorer** (`src/llm_eval/scorers/deterministic.py`)
   - JSON parsing with fallback strategies
   - Exact match scoring
   - Parse success tracking
   - Score aggregation

5. **SQLite Storage** (`src/llm_eval/metrics/storage.py`)
   - Two-table schema (runs + sample_results)
   - Run metadata persistence
   - Sample-level result storage
   - Query helpers (get_run, list_runs)

6. **Metrics Collector** (`src/llm_eval/metrics/collector.py`)
   - `EvaluationResult` dataclass
   - Run ID generation (timestamp + UUID)
   - Metric aggregation (accuracy, parse_rate)
   - Failed sample filtering

7. **Evaluation Runner** (`src/llm_eval/runner.py`)
   - End-to-end orchestration
   - Task → Model → Scorer → Storage flow
   - Progress logging
   - Batch processing

8. **CLI Script** (`scripts/run_eval.py`)
   - Argparse interface
   - Model + task selection
   - Sample limit control
   - Temperature/max_tokens config

9. **Model Config** (`configs/models.yaml`)
   - OpenAI model presets
   - Anthropic placeholders
   - Environment variable substitution

### Testing

**Mock Testing** (`scripts/test_pipeline.py`):
- Component-level tests (6 tests)
- Full pipeline test with mock model
- No API key required
- All tests passing ✅

**Results**:
```
Testing components...
1. ✓ Loaded 120 test samples
2. ✓ Prompt formatting (2 messages)
3. ✓ Mock model generation
4. ✓ JSON scoring (parse + exact match)
5. ✓ Metrics collection & aggregation
6. ✓ SQLite storage read/write

Testing full pipeline...
✓ 3-sample end-to-end test passed
```

## Dataset Status

**JSON Extraction**: 2,150 samples
- Train: 2,000 samples
- Validation: 30 samples
- Test: 120 samples

**Structure**: Each sample has:
- `id`: Unique identifier
- `input`: Text to extract from
- `expected_output`: Ground truth JSON
- `schema`: JSON Schema definition
- `slice`: Categories (e.g., "hard", "nested_object")
- `difficulty`: easy/medium/hard

## Usage

### Run Evaluation
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Full test set (120 samples)
python3 scripts/run_eval.py --task json --model gpt-4o-mini

# Small batch (2 samples)
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 2
```

### Test Without API
```bash
python3 scripts/test_pipeline.py
```

## Technical Decisions

1. **SQLite over files**: Structured queries, concurrent writes, easy scaling
2. **Sync over async**: Simplicity first, can parallelize later
3. **JSONL format**: Line-by-line streaming, easy debugging
4. **Dataclasses**: Type safety without Pydantic dependency
5. **Logging**: Console-first, can add file/remote later

## Files Created This Session

```
src/llm_eval/models/base.py         (97 lines)
src/llm_eval/models/api_model.py    (70 lines)
src/llm_eval/tasks/json_extraction.py (102 lines)
src/llm_eval/scorers/deterministic.py (107 lines)
src/llm_eval/metrics/storage.py     (205 lines)
src/llm_eval/metrics/collector.py   (144 lines)
src/llm_eval/runner.py              (127 lines)
scripts/run_eval.py                 (118 lines)
scripts/test_pipeline.py            (165 lines)
configs/models.yaml                 (28 lines)
M1_COMPLETE.md                      (documentation)
```

**Total**: ~1,163 lines of production code + tests

## Known Limitations

1. **No async**: Sequential API calls (slow for large batches)
2. **No retry logic**: API failures not handled
3. **Single task**: Only JSON extraction implemented
4. **Basic metrics**: Only exact match + parse rate
5. **No caching**: Every run hits the API

## Next Session (M2)

Ready to build:
1. QA task (question answering)
2. Code generation task
3. More model backends (Anthropic, vLLM, local models)
4. Async batch processing with rate limiting
5. Advanced metrics (F1, BLEU, perplexity)
6. Web UI for result browsing
7. Slice-based analysis (difficulty, category)

## Git Status

```bash
git log --oneline -1
# Initial commit with project structure

# Ready to commit:
git add src/ scripts/ configs/ M1_COMPLETE.md sessions/session2.md
git commit -m "feat: complete M1 evaluation pipeline

- Implement base model interface + OpenAI API
- Add JSON extraction task with 2,150 samples
- Build deterministic scorer with exact match
- Create SQLite storage for metrics
- Add evaluation runner + CLI
- Test full pipeline with mock model

Closes M1. Ready for M2 (QA + Code tasks)."
```

## Session Stats

- **Duration**: ~1 hour
- **Components**: 9/9 complete
- **Tests**: All passing
- **Lines**: ~1,163 total
- **Commits**: Ready for 1st M1 commit

---

**Status**: M1 complete, tested, documented ✅  
**Next**: Continue with M2 task expansion or test with real API
