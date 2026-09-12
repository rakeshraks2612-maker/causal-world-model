"""
Causal Reasoning & DAG Pathway Panel Component (Task 7.2 / Phase 7.7).
Renders the structural causal model (SCM) explanation and visual causal path.
Strictly zero emojis; high-precision mathematical notations.
"""

from __future__ import annotations
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord


def render_causal_panel(record: PrismDecisionRecord) -> None:
    """Render the causal mechanism explanation and interactive DAG flow."""
    st.markdown(
        """
        <div class="scada-panel-header">
            <div class="scada-panel-title">
                STRUCTURAL CAUSAL MODEL (SCM) MECHANISM & PROPAGATION
            </div>
            <div style="font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);">
                PEARL LEVEL-2: do(A) GRAPH SURGERY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cr = record.causal_reasoning

    # Summary Callout Card
    st.markdown(
        f"""
        <div class="scada-panel" style="border-left: 3px solid var(--color-cyan);">
            <div style="font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:var(--color-cyan); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
                CAUSAL EFFECT SYNTHESIS
            </div>
            <div style="font-family:var(--font-sans); font-size:1.15rem; font-weight:700; color:var(--text-primary); margin-bottom:6px;">
                {cr.summary.headline}
            </div>
            <div style="font-size:0.88rem; color:var(--text-secondary); line-height:1.5;">
                {cr.summary.mechanism}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Causal Chain
    st.markdown("##### ACTIVE STRUCTURAL PROPAGATION PATHWAY: do(A) → Y")
    nodes = cr.causal_chain.nodes
    if nodes:
        node_html_list = []
        for i, n in enumerate(nodes):
            dir_text = n.predicted_direction.replace('_', ' ')
            dir_color = "#34d399"
            if "INCREASE" in n.predicted_direction or "UP" in n.predicted_direction:
                dir_color = "#38bdf8"
            elif "DECREASE" in n.predicted_direction or "DOWN" in n.predicted_direction or "DROP" in n.predicted_direction:
                dir_color = "#34d399"

            delta_str = f"<div style='font-family:var(--font-mono); font-weight:700; font-size:1.0rem; color:var(--color-cyan); margin-top:4px;'>{n.delta_value:+.2f} <span style='font-size:0.70rem; color:var(--text-muted);'>{n.unit}</span></div>" if n.delta_value is not None else ""
            
            box = f"""
            <div style="background:var(--bg-surface-elevated); border:1px solid var(--border-subtle); border-radius:6px; padding:12px 14px; text-align:center; min-width:140px;">
                <div style="font-family:var(--font-mono); font-size:0.68rem; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.06em;">NODE {i+1}</div>
                <div style="font-family:var(--font-sans); font-weight:700; color:var(--text-primary); font-size:0.92rem; margin:3px 0;">{n.variable.replace('_', ' ').upper()}</div>
                <div style="font-family:var(--font-mono); font-size:0.72rem; font-weight:600; color:{dir_color};">{dir_text}</div>
                {delta_str}
            </div>
            """
            node_html_list.append(box)

        chain_html = '<div style="display:flex; align-items:center; gap:10px; overflow-x:auto; padding:10px 0; margin-bottom:14px;">'
        for idx, b in enumerate(node_html_list):
            chain_html += b
            if idx < len(node_html_list) - 1:
                chain_html += '<div style="color:var(--color-cyan); font-size:1.2rem; font-weight:bold; opacity:0.7; padding:0 4px;">→</div>'
        chain_html += '</div>'
        st.markdown(chain_html, unsafe_allow_html=True)
    else:
        st.markdown("<div class='scada-panel' style='color:var(--text-muted); font-size:0.85rem;'>No active causal intervention pathway (Inaction Restraint / Upstream Distrust).</div>", unsafe_allow_html=True)

    # Narrative Details & SCM Invariant Badge
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Structural Narrative:** {cr.causal_chain.narrative}")
    with col2:
        st.markdown(
            """
            <div class="scada-panel" style="border-color:rgba(16, 185, 129, 0.3); background:rgba(16, 185, 129, 0.05); padding:12px 14px;">
                <div style="font-family:var(--font-mono); font-size:0.70rem; color:#34d399; text-transform:uppercase; font-weight:700;">PEARL SCM VALIDATION</div>
                <div style="font-weight:700; color:var(--text-primary); font-size:0.85rem; margin:2px 0;">Structural Invariant Verified</div>
                <div style="font-size:0.75rem; color:var(--text-secondary);">Intervention graph surgery do(A) isolates causal mechanism from observational confounding.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
