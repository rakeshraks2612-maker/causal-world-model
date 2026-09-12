"""PRISM Dashboard Data Loader (Task 7.2).

Provides cached and direct scenario loading, connecting the dashboard
to the authoritative Task 7.1 PRISM Pipeline.

Architectural Invariant:
The data loader only invokes the pipeline or loads authoritative artifacts.
It contains zero independent planning, safety, or causal logic.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import numpy as np

from prism.pipeline.engine import PrismPipeline, PrismPipelineConfig
from prism.explanation.unified_record import PrismDecisionRecord
from prism.dataset.decision_benchmark import LearnerDecisionScenario


_GLOBAL_PIPELINE: Optional[PrismPipeline] = None


def get_pipeline(model_dir: str = "artifacts/baseline_005") -> PrismPipeline:
    """Singleton getter for the PRISM pipeline."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        cfg = PrismPipelineConfig(model_dir=Path(model_dir))
        _GLOBAL_PIPELINE = PrismPipeline(cfg)
    return _GLOBAL_PIPELINE


def load_benchmark_scenario(
    scenario_id: str,
    pipeline: Optional[PrismPipeline] = None,
    output_dir: Optional[str] = "artifacts/prism_runs",
    timestamp_utc: Optional[str] = None,
) -> Tuple[PrismDecisionRecord, Optional[LearnerDecisionScenario]]:
    """Execute or load a benchmark scenario via the authoritative pipeline."""
    pipe = pipeline or get_pipeline()
    bpath = Path(pipe.config.benchmark_dir) / "scenarios" / scenario_id
    learner_scen: Optional[LearnerDecisionScenario] = None

    if (bpath / "learner.npz").exists():
        learner_scen = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    record = pipe.run_scenario(scenario_id, output_dir=output_dir, timestamp_utc=timestamp_utc)
    return record, learner_scen


def load_telemetry_file(
    file_path: Union[str, Path],
    pipeline: Optional[PrismPipeline] = None,
    output_dir: Optional[str] = "artifacts/prism_runs",
    timestamp_utc: Optional[str] = None,
) -> Tuple[PrismDecisionRecord, Optional[LearnerDecisionScenario]]:
    """Execute a standalone telemetry .npz file via the pipeline."""
    pipe = pipeline or get_pipeline()
    fpath = Path(file_path)
    learner_scen = LearnerDecisionScenario.load_npz(fpath) if fpath.exists() else None
    record = pipe.run_file(fpath, output_dir=output_dir, timestamp_utc=timestamp_utc)
    return record, learner_scen


def load_saved_record_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a raw JSON decision record artifact for audit inspection."""
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Decision record artifact not found at {p}")
    with open(p, "r") as f:
        return json.load(f)


def load_saved_dossier_markdown(md_path: Union[str, Path]) -> str:
    """Load a raw Markdown dossier artifact for audit inspection."""
    p = Path(md_path)
    if not p.exists():
        raise FileNotFoundError(f"Markdown dossier artifact not found at {p}")
    with open(p, "r") as f:
        return f.read()
