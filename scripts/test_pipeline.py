#!/usr/bin/env python3
"""Test script to verify pipeline structure without API calls."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_eval.models.base import BaseModel, Message, ModelResponse
from llm_eval.tasks.json_extraction import JSONExtractionTask
from llm_eval.scorers.deterministic import score_json_extraction
from llm_eval.metrics.collector import MetricsCollector
from llm_eval.metrics.storage import MetricsStorage
from llm_eval.runner import EvaluationRunner


class MockModel(BaseModel):
    """Mock model for testing."""

    def generate(self, messages, temperature=0.0, max_tokens=None, **kwargs):
        # Return a simple JSON response
        return ModelResponse(
            content='{"test": "value"}',
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )

    def batch_generate(self, batch_messages, temperature=0.0, max_tokens=None, **kwargs):
        return [self.generate(msgs, temperature, max_tokens, **kwargs) for msgs in batch_messages]


def test_components():
    """Test individual components."""
    print("Testing components...\n")

    # Test 1: Task loading
    print("1. Testing task loading...")
    project_root = Path(__file__).parent.parent
    task = JSONExtractionTask(
        data_path=str(project_root / "data" / "tasks" / "json_extraction"),
        split="test"
    )
    print(f"   ✓ Loaded {len(task)} samples")
    sample = task.get_samples(limit=1)[0]
    print(f"   ✓ Sample ID: {sample.id}")

    # Test 2: Prompt formatting
    print("\n2. Testing prompt formatting...")
    messages = task.format_prompt(sample)
    print(f"   ✓ Generated {len(messages)} messages")
    print(f"   ✓ System prompt: {messages[0].content[:50]}...")

    # Test 3: Mock model
    print("\n3. Testing mock model...")
    model = MockModel("mock-model-v1")
    response = model.generate(messages)
    print(f"   ✓ Model: {response.model}")
    print(f"   ✓ Response: {response.content}")

    # Test 4: Scoring
    print("\n4. Testing scorer...")
    # Use a matching expected output for testing
    scores = score_json_extraction('{"test": "value"}', {"test": "value"})
    print(f"   ✓ Parse success: {scores['parse_success']}")
    print(f"   ✓ Exact match: {scores['exact_match']}")

    # Test 5: Metrics collection
    print("\n5. Testing metrics collector...")
    collector = MetricsCollector()
    collector.add_result(
        sample_id="test_001",
        input_text="test input",
        expected_output={"test": "value"},
        model_response='{"test": "value"}',
        scores=scores
    )
    print(f"   ✓ Run ID: {collector.run_id}")
    print(f"   ✓ Results collected: {len(collector.results)}")
    agg = collector.aggregate()
    print(f"   ✓ Aggregated metrics: {agg}")

    # Test 6: Storage
    print("\n6. Testing storage...")
    storage = MetricsStorage("data/metrics/test.db")
    storage.save_run(
        run_id=collector.run_id,
        model_name="mock-model",
        task_name="json",
        config={"test": True},
        aggregate_metrics=agg
    )
    print(f"   ✓ Saved run to database")
    retrieved = storage.get_run(collector.run_id)
    print(f"   ✓ Retrieved run: {retrieved['model_name']}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


def test_full_pipeline():
    """Test full pipeline with mock model."""
    print("\n\nTesting full pipeline with mock model...\n")

    project_root = Path(__file__).parent.parent
    model = MockModel("mock-model-v1")

    runner = EvaluationRunner(
        model=model,
        task_name="json",
        data_path=str(project_root / "data" / "tasks" / "json_extraction"),
        storage_path="data/metrics/test.db"
    )

    results = runner.run(limit=3)

    print("\n" + "="*60)
    print("PIPELINE TEST RESULTS")
    print("="*60)
    print(f"Run ID: {results['run_id']}")
    print(f"Model: {results['model']}")
    print(f"Task: {results['task']}")
    print(f"Samples: {results['sample_count']}")
    print("\nMetrics:")
    for key, value in results['aggregate_metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    test_components()
    test_full_pipeline()
    print("\n✓ All tests completed successfully!")
