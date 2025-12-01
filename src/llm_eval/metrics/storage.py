"""SQLite storage for evaluation metrics."""
import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class MetricsStorage:
    """SQLite-based storage for evaluation metrics."""

    def __init__(self, db_path: str = "data/metrics/eval_results.db"):
        """
        Initialize metrics storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    config TEXT,
                    aggregate_metrics TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sample_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    input TEXT NOT NULL,
                    expected_output TEXT,
                    model_response TEXT,
                    scores TEXT,
                    metadata TEXT,
                    FOREIGN KEY (run_id) REFERENCES eval_runs (run_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_id
                ON sample_results (run_id)
            """)

            conn.commit()

    def save_run(
        self,
        run_id: str,
        model_name: str,
        task_name: str,
        config: Dict[str, Any],
        aggregate_metrics: Dict[str, Any]
    ):
        """
        Save evaluation run metadata.

        Args:
            run_id: Unique run identifier
            model_name: Name of the model evaluated
            task_name: Name of the task
            config: Run configuration
            aggregate_metrics: Aggregated metrics for the run
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                (run_id, model_name, task_name, timestamp, config, aggregate_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_name,
                    task_name,
                    datetime.now().isoformat(),
                    json.dumps(config),
                    json.dumps(aggregate_metrics)
                )
            )
            conn.commit()

    def save_sample_result(
        self,
        run_id: str,
        sample_id: str,
        input_text: str,
        expected_output: Any,
        model_response: str,
        scores: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save individual sample result.

        Args:
            run_id: Run identifier
            sample_id: Sample identifier
            input_text: Input text
            expected_output: Expected output
            model_response: Model's response
            scores: Score metrics
            metadata: Additional metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sample_results
                (run_id, sample_id, input, expected_output, model_response, scores, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sample_id,
                    input_text,
                    json.dumps(expected_output),
                    model_response,
                    json.dumps(scores),
                    json.dumps(metadata) if metadata else None
                )
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve run metadata.

        Args:
            run_id: Run identifier

        Returns:
            Run metadata or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM eval_runs WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    "run_id": row["run_id"],
                    "model_name": row["model_name"],
                    "task_name": row["task_name"],
                    "timestamp": row["timestamp"],
                    "config": json.loads(row["config"]),
                    "aggregate_metrics": json.loads(row["aggregate_metrics"])
                }

        return None

    def get_sample_results(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all sample results for a run.

        Args:
            run_id: Run identifier

        Returns:
            List of sample results
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM sample_results WHERE run_id = ?",
                (run_id,)
            )

            results = []
            for row in cursor.fetchall():
                results.append({
                    "sample_id": row["sample_id"],
                    "input": row["input"],
                    "expected_output": json.loads(row["expected_output"]),
                    "model_response": row["model_response"],
                    "scores": json.loads(row["scores"]),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                })

            return results

    def list_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent evaluation runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT run_id, model_name, task_name, timestamp, aggregate_metrics
                FROM eval_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            )

            runs = []
            for row in cursor.fetchall():
                runs.append({
                    "run_id": row["run_id"],
                    "model_name": row["model_name"],
                    "task_name": row["task_name"],
                    "timestamp": row["timestamp"],
                    "aggregate_metrics": json.loads(row["aggregate_metrics"])
                })

            return runs
