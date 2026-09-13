"""
PRISM Decision Intelligence Streamlit Application (Task 7.2 / Phase 7.7).

Interactive Decision Intelligence Console for PRISM.
Renders the high-performance, cyber-physical mission control interface
inspired by KuberMesh, Linear, and Palantir Foundry.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import tempfile
import streamlit as st
import streamlit.components.v1 as components

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism.dashboard.theme import get_custom_css
from prism.dashboard.data_loader import (
    load_benchmark_scenario,
    load_telemetry_file,
    get_pipeline,
)
from prism.dashboard.spa_renderer import build_dashboard_html


def configure_page() -> None:
    """Set up Streamlit layout and custom design system."""
    favicon_path = REPO_ROOT / "prism" / "dashboard" / "web" / "favicon.png"
    st.set_page_config(
        page_title="PRISM · Superintelligence for physical judgment",
        page_icon=str(favicon_path) if favicon_path.exists() else "💎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #05070c !important;
            }
            .block-container {
                padding: 0rem 0.5rem !important;
                max-width: 100% !important;
            }
            header[data-testid="stHeader"] {
                display: none !important;
            }
            footer {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_cached_scenarios_bundle():
    """Precompute and cache all 6 frozen benchmark scenario records."""
    pipeline = get_pipeline()
    scenarios_dict = {}
    benchmark_keys = [
        "scenario_04_pump",
        "scenario_01_do_nothing",
        "scenario_02_valve",
        "scenario_03_throttle",
        "scenario_05_combined",
        "scenario_06_all_unsafe",
    ]

    for scen_key in benchmark_keys:
        rec, l = load_benchmark_scenario(scen_key, pipeline=pipeline)
        t_star = l.intervention_time if l is not None and l.historical_observations is not None else 0
        obs = l.historical_observations[t_star] if l is not None and l.historical_observations is not None else [91.4, 68.2, 4.1, 31.7, 87.0, 0, 0, 4.7]
        acts = l.historical_actions[t_star] if l is not None and l.historical_actions is not None else [62.0, 2.0]
        
        rec_dict = rec.to_dict()
        rec_dict["telemetry_snapshot"] = {
            "t_core": float(obs[0]),
            "t_cool": float(obs[1]),
            "p_sys": float(obs[2]),
            "f_cool": float(obs[3]),
            "l_cpu": float(obs[4]),
            "v_pos": float(acts[0] * 100.0 if acts[0] <= 1.0 else acts[0]),
            "p_speed": float(acts[1]),
            "d_lat": float(rec.trust.latent_novelty_d),
        }
        rec_dict["markdown_dossier"] = rec.format_markdown()
        rec_dict["raw_record"] = rec.to_dict()
        scenarios_dict[scen_key] = rec_dict

    return scenarios_dict


def main() -> None:
    configure_page()

    # Sidebar: Optional Custom Telemetry Upload
    st.sidebar.markdown(
        """
        <div style="font-family:'Geist Mono', monospace; font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:8px;">
            TELEMETRY INGESTION (OPTIONAL)
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.sidebar.file_uploader("Upload .npz Telemetry File", type=["npz"])

    scenarios_bundle = get_cached_scenarios_bundle().copy()
    initial_key = "scenario_04_pump"

    if uploaded_file is not None:
        try:
            pipeline = get_pipeline()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            rec, l = load_telemetry_file(tmp_path, pipeline=pipeline)
            t_star = l.intervention_time if l is not None and l.historical_observations is not None else 0
            obs = l.historical_observations[t_star] if l is not None and l.historical_observations is not None else [91.4, 68.2, 4.1, 31.7, 87.0, 0, 0, 4.7]
            acts = l.historical_actions[t_star] if l is not None and l.historical_actions is not None else [62.0, 2.0]
            
            rec_dict = rec.to_dict()
            rec_dict["telemetry_snapshot"] = {
                "t_core": float(obs[0]),
                "t_cool": float(obs[1]),
                "p_sys": float(obs[2]),
                "f_cool": float(obs[3]),
                "l_cpu": float(obs[4]),
                "v_pos": float(acts[0] * 100.0 if acts[0] <= 1.0 else acts[0]),
                "p_speed": float(acts[1]),
                "d_lat": float(rec.trust.latent_novelty_d),
            }
            rec_dict["markdown_dossier"] = rec.format_markdown()
            rec_dict["raw_record"] = rec.to_dict()
            scenarios_bundle["custom_upload"] = rec_dict
            initial_key = "custom_upload"
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to process uploaded file: {e}")

    # Build and Render Full Standalone Cyber-Physical Mission Control UI
    html_content = build_dashboard_html(scenarios_bundle, initial_scenario_key=initial_key)
    components.html(html_content, height=1320, scrolling=True)


if __name__ == "__main__":
    main()
