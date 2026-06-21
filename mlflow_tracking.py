"""
mlflow_tracking.py
───────────────────
MLflow wrapper de tracking experiments.

Usage:
    from mlflow_tracking import MLflowTracker
    tracker = MLflowTracker(experiment_name="deep-learning-cars")
    tracker.log_params({"lr": 0.0003, "gamma": 0.99})
    tracker.log_metrics({"reward": 100.5, "loss": 0.02}, step=10)
    tracker.save_model("checkpoints/best.pt")
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker:
    """
    Wrapper tich hop MLflow de tracking experiments.

    Neu MLflow chua duoc cai, se fallback sang print log.
    """

    def __init__(
        self,
        experiment_name: str = "deep-learning-cars",
        tracking_uri: str = None,
        run_name: str = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and MLFLOW_AVAILABLE
        self._run = None

        if self.enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._run = mlflow.start_run(run_name=run_name)
            print(f"  [MLflow] Experiment: {experiment_name}")
            print(f"  [MLflow] Run ID: {self._run.info.run_id}")
        elif not MLFLOW_AVAILABLE and enabled:
            print("  [MLflow] WARNING: mlflow not installed, tracking disabled.")
            print("  [MLflow] Install with: pip install mlflow")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics (fitness, reward, loss, etc.)."""
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int = None) -> None:
        """Log single metric."""
        if self.enabled:
            mlflow.log_metric(key, value, step=step)

    def save_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Luu artifact (model weights, plots, etc.)."""
        if self.enabled and Path(local_path).exists():
            mlflow.log_artifact(local_path, artifact_path)

    def save_model(self, model_path: str) -> None:
        """Luu model file lam artifact."""
        self.save_artifact(model_path, "models")

    def end_run(self) -> None:
        """Ket thuc MLflow run."""
        if self.enabled and self._run:
            mlflow.end_run()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()

    def __del__(self):
        self.end_run()
