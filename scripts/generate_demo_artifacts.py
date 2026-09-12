"""Generate Frozen Demo Scenarios & Expected Artifacts (Task 7.3).

Executes the pipeline with fixed timestamps across the three golden demo scenarios:
1. demo_01_decision (S4 - Pump Modulation)
2. demo_02_uncertainty (S2 - Valve Intervention / Safety Catch)
3. demo_03_abstention (S6 - Upfront Distrust / Model Abstain)

Saves machine-readable JSON records and Markdown audit dossiers in demo/scenarios/
and frozen expected copies in demo/expected/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from prism.pipeline.engine import PrismPipeline, PrismPipelineConfig

FIXED_DEMO_TIMESTAMP = "2026-09-11T12:00:00Z"


def generate_demo_artifacts():
    pipeline = PrismPipeline(PrismPipelineConfig(model_dir=Path("artifacts/baseline_005")))
    demo_dir = Path("demo")
    scen_dir = demo_dir / "scenarios"
    exp_dir = demo_dir / "expected"
    scen_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    demos = [
        ("demo_01_decision", "scenario_04_pump"),
        ("demo_02_uncertainty", "scenario_02_valve"),
        ("demo_03_abstention", "scenario_06_all_unsafe"),
    ]

    for demo_name, scen_id in demos:
        record = pipeline.run_scenario(
            scenario_id=scen_id,
            timestamp_utc=FIXED_DEMO_TIMESTAMP,
        )

        # Save to demo/scenarios/
        scen_json = scen_dir / f"{demo_name}.json"
        scen_md = scen_dir / f"{demo_name}.md"
        with open(scen_json, "w") as f:
            f.write(record.to_json(indent=2))
        with open(scen_md, "w") as f:
            f.write(record.format_markdown())

        # Save to demo/expected/
        exp_json = exp_dir / f"{demo_name}_expected.json"
        with open(exp_json, "w") as f:
            f.write(record.to_json(indent=2))

        print(f"Generated {demo_name} ({scen_id}) -> SHA-256: {record.provenance.unified_record_hash}")


if __name__ == "__main__":
    generate_demo_artifacts()
