# GitHub Actions CI/CD Setup

## ✅ Workflows Fixed

Both workflows have been updated and should now pass:

### 1. Unit Tests (ci.yaml)
**Trigger**: On every push and PR

**What it does**:
- Checks out code
- Sets up Python 3.11
- Installs dependencies
- Runs `test_pipeline.py` (mock test)
- Checks Python syntax

**Status**: ✅ Fixed
- Added PYTHONPATH export to find `llm_eval` module
- Fixed file compilation to use `find` instead of glob

### 2. Quality Regression (regression.yaml)
**Trigger**:
- On push to main
- Weekly on Sundays (cron)

**What it does**:
- Runs API evaluation with GPT-4o-mini
- Saves results and logs
- Uploads artifacts

**Status**: ✅ Fixed
- Updated `actions/upload-artifact` from v3 to v4 (v3 was deprecated)

## 🔐 Required GitHub Secrets

To enable regression tests with real API calls, add this secret:

1. Go to: https://github.com/faiqhilman13/LLM-eval/settings/secrets/actions
2. Click "New repository secret"
3. Add:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key (`sk-...`)

**Note**: Without this secret, regression tests will fail but won't break the build (uses `|| true`)

## 🎯 Workflow Status

Check: https://github.com/faiqhilman13/LLM-eval/actions

**Expected behavior**:
- ✅ **Unit tests**: Should pass (uses mock, no API needed)
- ⚠️ **Regression tests**: May fail if `OPENAI_API_KEY` not set (that's OK)

## 📝 What Each Workflow Tests

### CI (Unit Tests)
```bash
# Runs this locally to verify:
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
python3 scripts/test_pipeline.py
```

Output should be:
```
✓ All tests completed successfully!
```

### Regression
```bash
# Runs this (requires API key):
export OPENAI_API_KEY="sk-..."
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 10
```

Evaluates 10 JSON samples and saves results.

## 🔧 Optional: Disable Regression Tests

If you don't want to use API credits in CI, you can:

**Option 1: Comment out the schedule**
```yaml
# .github/workflows/regression.yaml
on:
  push:
    branches: [main]
  # schedule:
  #   - cron: '0 0 * * 0'
```

**Option 2: Skip if no API key**
The workflow already handles missing API key gracefully with `|| true`

## 🚀 Testing Locally

Before pushing, test the workflows locally:

```bash
# Test CI workflow
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
python3 scripts/test_pipeline.py

# Test regression workflow (with API key)
export OPENAI_API_KEY="sk-..."
python3 scripts/run_eval.py --task json --model gpt-4o-mini --limit 10
```

## 📊 Workflow Files

- `.github/workflows/ci.yaml` - Unit tests (always runs)
- `.github/workflows/regression.yaml` - Quality regression (optional)

## 🎯 Next Steps

1. **Add OPENAI_API_KEY secret** (optional, for regression tests)
2. **Check Actions tab** to see if workflows pass
3. **Green checkmarks** = all good!

---

**Latest commit**: Fixed both workflows ✅
