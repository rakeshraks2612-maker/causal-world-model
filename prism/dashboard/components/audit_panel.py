"""
Audit & Cryptographic Provenance Panel Component (Task 7.2 / Phase 7.7).
Renders the immutable audit trail and deterministic SHA-256 fingerprint.
Strictly zero emojis; cryptographic provenance and reproducibility console.
"""

from __future__ import annotations
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord


def render_audit_panel(record: PrismDecisionRecord) -> None:
    """Render the cryptographic audit trail and artifact viewers."""
    st.markdown(
        """
        <div class="scada-panel-header">
            <div class="scada-panel-title">
                CRYPTOGRAPHIC PROVENANCE & TAMPER-EVIDENT AUDIT DOSSIER
            </div>
            <div style="font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);">
                CANONICAL JSON RECORD // DETERMINISTIC HASH
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prov = record.provenance

    # Root Fingerprint Highlight Box
    st.markdown(
        f"""
        <div class="scada-panel" style="border: 1px solid rgba(6, 182, 212, 0.3); background: #080c14;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div style="font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:var(--color-cyan); letter-spacing:0.06em; text-transform:uppercase;">
                    DETERMINISTIC SHA-256 EVIDENCE FINGERPRINT
                </div>
                <span class="status-tag status-tag-cyan">
                    TAMPER-EVIDENT
                </span>
            </div>
            <div class="hash-display">
                {prov.unified_record_hash}
            </div>
            <div style="font-size:0.78rem; color:var(--text-muted); margin-top:4px;">
                Every telemetry sample, causal DAG node, counterfactual rollout, and safety headroom margin is sealed in this 256-bit cryptographic digest. Any post-hoc mutation invalidates this fingerprint.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sub-Engine Metadata Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">MODEL CHECKPOINT</div>
                <div style="font-family:var(--font-mono); font-size:0.88rem; color:var(--text-primary); font-weight:600;">{prov.model_version}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">PLANNER ENGINE</div>
                <div style="font-family:var(--font-mono); font-size:0.88rem; color:var(--text-primary); font-weight:600;">{prov.planner_version}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">DATASET PROTOCOL</div>
                <div style="font-family:var(--font-mono); font-size:0.88rem; color:var(--text-primary); font-weight:600;">{prov.dataset_version}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">TIMESTAMP (UTC)</div>
                <div style="font-family:var(--font-mono); font-size:0.88rem; color:var(--text-primary); font-weight:600;">{prov.timestamp_utc[:19]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Download Buttons
    json_str = record.to_json(indent=2)
    md_str = record.format_markdown()
    scen_name = prov.scenario_id or "prism_decision"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            label="DOWNLOAD CANONICAL JSON RECORD",
            data=json_str,
            file_name=f"{scen_name}_decision_record.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_btn2:
        st.download_button(
            label="DOWNLOAD MARKDOWN AUDIT DOSSIER",
            data=md_str,
            file_name=f"{scen_name}_audit_dossier.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Expanders
    with st.expander("INSPECT FULL MACHINE-READABLE JSON RECORD"):
        st.json(record.to_dict())

    with st.expander("INSPECT FORMATTED EXECUTIVE MARKDOWN DOSSIER"):
        st.markdown(md_str)
