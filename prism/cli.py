"""PRISM Command-Line Interface (Task 7.1).

Deterministic, production-ready CLI entrypoint for PRISM:
Takes a scenario ID, episode file, or live telemetry stream and produces
the complete PRISM decision + evidence + audit dossier end-to-end.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

from prism.pipeline.engine import PrismPipeline, PrismPipelineConfig
from prism.explanation.unified_record import PrismDecisionRecord


def format_terminal_banner(record: PrismDecisionRecord) -> str:
    """Format a styled executive ASCII terminal summary for judges and operators."""
    scen_id = record.provenance.scenario_id or "live_telemetry"
    status = record.decision.decision_status
    rec = record.decision.recommendation or "NONE (ABSTAINED)"
    trust = record.trust.trust_state
    safety_state = record.safety.overall_state
    is_safe_sym = "✅ PASS" if record.safety.is_safe else "❌ FAIL"
    if record.safety.overall_state == "MARGINAL":
        is_safe_sym = "⚠️ MARGINAL (COMPLIANT)"
    elif record.safety.overall_state == "ABSTAIN_REQUIRED":
        is_safe_sym = "🛑 ABSTAIN REQUIRED"

    lim_c = record.safety.limiting_constraint
    lim_str = f"{lim_c.constraint_name.title()} ({lim_c.variable_symbol})"
    margin_str = f"{lim_c.raw_margin:+.2f} {lim_c.unit}" if lim_c.raw_margin is not None else "N/A"
    
    rt_str = f"{record.trust.reconstruction_residual_t_core:.2f}°C"
    r8d_str = f"{record.trust.reconstruction_residual_8d:.2f}"
    d_lat_str = f"{record.trust.latent_novelty_d:.2f} d_M"
    u_str = f"{record.decision_quality.utility_score:.4f}" if record.decision_quality.utility_score is not None else "N/A"
    hash_str = record.provenance.unified_record_hash

    cf_summary = "N/A"
    if record.counterfactual is not None:
        cf = record.counterfactual
        dt = f"{cf.causal_effect.delta_t_core_peak:+.2f}°C" if cf.causal_effect.delta_t_core_peak is not None else "N/A"
        df = f"{cf.causal_effect.delta_f_cool_min:+.2f} L/min" if cf.causal_effect.delta_f_cool_min is not None else "N/A"
        cf_summary = f"do({cf.intervention.intervention_target}={cf.intervention.counterfactual_value:.1f}) -> ΔT_peak={dt}, ΔF_min={df}"

    banner = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║                     🏛️  PRISM DECISION INTELLIGENCE ENGINE                   ║",
        "╠══════════════════════════════════════════════════════════════════════════════╣",
        f"║ Scenario ID:         {scen_id:<55} ║",
        f"║ Decision Status:     {status:<55} ║",
        f"║ Recommendation:      {rec:<55} ║",
        f"║ Trust State:         {trust:<55} ║",
        f"║ Residuals:           R_T = {rt_str:<10} | R_8D = {r8d_str:<8} | D_lat = {d_lat_str:<12} ║",
        f"║ Safety Status:       {is_safe_sym:<55} ║",
        f"║ Limiting Constraint: {lim_str:<25} Headroom Margin: {margin_str:<18} ║",
        f"║ Multi-Obj Utility:   {u_str:<55} ║",
        f"║ Counterfactual:      {cf_summary:<55} ║",
        "╠══════════════════════════════════════════════════════════════════════════════╣",
        f"║ SHA-256 Fingerprint: {hash_str:<55} ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
    ]
    return "\n".join(banner)


def parse_scenario_alias(scen_arg: str) -> str:
    """Support short aliases like s1, s2, s6, or full names."""
    alias_map = {
        "s1": "scenario_01_do_nothing",
        "s2": "scenario_02_valve",
        "s3": "scenario_03_throttle",
        "s4": "scenario_04_pump",
        "s5": "scenario_05_combined",
        "s6": "scenario_06_all_unsafe",
        "1": "scenario_01_do_nothing",
        "2": "scenario_02_valve",
        "3": "scenario_03_throttle",
        "4": "scenario_04_pump",
        "5": "scenario_05_combined",
        "6": "scenario_06_all_unsafe",
        "scenario_1": "scenario_01_do_nothing",
        "scenario_2": "scenario_02_valve",
        "scenario_3": "scenario_03_throttle",
        "scenario_4": "scenario_04_pump",
        "scenario_5": "scenario_05_combined",
        "scenario_6": "scenario_06_all_unsafe",
    }
    return alias_map.get(scen_arg.lower(), scen_arg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM Decision & Evidence Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario", "-s",
        type=str,
        help="Benchmark scenario name or alias (e.g. s1, s2, scenario_06_all_unsafe)",
    )
    group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to telemetry .npz file",
    )
    group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Execute all 6 benchmark scenarios end-to-end",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="artifacts/prism_runs",
        help="Directory to save JSON record and Markdown audit dossier",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts/baseline_005",
        help="Directory containing frozen world model artifacts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON record to stdout",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output full Markdown dossier to stdout",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress terminal banner",
    )

    args = parser.parse_args()

    cfg = PrismPipelineConfig(
        model_dir=Path(args.model_dir),
    )
    pipeline = PrismPipeline(cfg)

    if args.all:
        print(f"\n🚀 Executing complete PRISM benchmark suite across all 6 scenarios...")
        t0 = time.perf_counter()
        results = pipeline.run_all_benchmark_scenarios(output_dir=args.output_dir)
        total_time = time.perf_counter() - t0
        
        print(f"✅ Completed 6/6 scenarios in {total_time:.2f}s.")
        print(f"📁 Dossiers saved to: {args.output_dir}/\n")

        for s_id, record in results.items():
            print(format_terminal_banner(record))
            print()
        sys.exit(0)

    elif args.scenario:
        scen_id = parse_scenario_alias(args.scenario)
        record = pipeline.run_scenario(scen_id, output_dir=args.output_dir)

    elif args.input:
        record = pipeline.run_file(args.input, output_dir=args.output_dir)

    if args.json:
        print(record.to_json(indent=2))
    elif args.markdown:
        print(record.format_markdown())
    elif not args.quiet:
        print(format_terminal_banner(record))
        print(f"\n📁 Saved audit records to:")
        print(f"  - JSON: {args.output_dir}/{(record.provenance.scenario_id or 'telemetry')}_decision_record.json")
        print(f"  - Dossier: {args.output_dir}/{(record.provenance.scenario_id or 'telemetry')}_audit_dossier.md\n")


if __name__ == "__main__":
    main()
