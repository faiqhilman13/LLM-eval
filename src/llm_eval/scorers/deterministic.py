"""Deterministic scoring functions for evaluation."""
import json
from typing import Any, Dict, Tuple


def parse_json_response(response: str) -> Tuple[Any, bool]:
    """
    Parse JSON from model response.

    Args:
        response: Raw model response text

    Returns:
        Tuple of (parsed_json, success_flag)
    """
    # Try to extract JSON from markdown code blocks
    if "```json" in response:
        try:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
            return json.loads(json_str), True
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to parse entire response as JSON
    try:
        return json.loads(response.strip()), True
    except json.JSONDecodeError:
        pass

    # Try to find JSON object or array in the response
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = response.find(start_char)
        end_idx = response.rfind(end_char)
        if start_idx != -1 and end_idx != -1:
            try:
                json_str = response[start_idx:end_idx + 1]
                return json.loads(json_str), True
            except json.JSONDecodeError:
                continue

    return None, False


def exact_match_score(predicted: Any, expected: Any) -> float:
    """
    Calculate exact match score.

    Args:
        predicted: Model prediction
        expected: Ground truth

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if predicted == expected else 0.0


def score_json_extraction(response: str, expected: Any) -> Dict[str, Any]:
    """
    Score JSON extraction task.

    Args:
        response: Raw model response
        expected: Expected JSON output

    Returns:
        Dictionary with scoring metrics
    """
    parsed, parse_success = parse_json_response(response)

    if not parse_success:
        return {
            "exact_match": 0.0,
            "parse_success": False,
            "error": "Failed to parse JSON from response"
        }

    exact_match = exact_match_score(parsed, expected)

    return {
        "exact_match": exact_match,
        "parse_success": True,
        "parsed_output": parsed,
    }


def aggregate_scores(scores: list[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate scores across multiple samples.

    Args:
        scores: List of individual score dictionaries

    Returns:
        Aggregated metrics
    """
    if not scores:
        return {}

    total = len(scores)
    exact_matches = sum(s.get("exact_match", 0.0) for s in scores)
    parse_successes = sum(1 for s in scores if s.get("parse_success", False))

    return {
        "accuracy": exact_matches / total,
        "parse_rate": parse_successes / total,
        "total_samples": total,
    }
