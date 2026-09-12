"""
Counterfactual Twin-World Studio Panel Component (Task 7.2 / Phase 7.7).
Renders the Pearl Level-3 twin-world counterfactual comparison.
Strictly zero emojis; rigorous causal abduction and counterfactual metrics.
"""

from __future__ import annotations
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord


def render_counterfactual_panel(record: PrismDecisionRecord) -> None:
    """Render the retrospective twin-world counterfactual comparison."""
    st.markdown(
        """
        <div class="scada-panel-header">
            <div class="scada-panel-title">
                PEARL LEVEL-3 COUNTERFACTUAL STUDIO // TWIN-WORLD ARBITRATION
            </div>
            <div style="font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);">
                ABDUCTION → ACTION → PREDICTION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cf = record.counterfactual
    if not cf.available or cf.counterfactual_action is None:
        st.markdown(
            "<div class='scada-panel' style='color:var(--text-muted); font-size:0.85rem;'>"
            "Counterfactual evaluation withheld for this scenario (Inaction Restraint / Upstream Distrust)."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Abduction Mechanism Callout
    st.markdown(
        """
        <div class="scada-panel" style="background:rgba(6, 182, 212, 0.04); border-color:rgba(6, 182, 212, 0.25); padding:12px 16px; margin-bottom:14px;">
            <div style="font-family:var(--font-mono); font-size:0.70rem; font-weight:700; color:var(--color-cyan); text-transform:uppercase; letter-spacing:0.06em;">
                EXOGENOUS DISTURBANCE ABDUCTION (PEARL LEVEL-3)
            </div>
            <div style="font-size:0.84rem; color:var(--text-secondary); margin-top:3px; line-height:1.4;">
                Latent noise and unobserved exogenous disturbances $U_{1:T}$ were abduced from observed factual telemetry and replayed identically into the counterfactual twin world under $do(A_{\\text{alt}})$.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-Side Twin Worlds
    col_fact, col_cf = st.columns(2)

    with col_fact:
        st.markdown(
            f"""
            <div class="scada-panel" style="border-top:2px solid var(--color-slate);">
                <div style="font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">
                    FACTUAL WORLD (OBSERVED HISTORICAL)
                </div>
                <div style="font-family:var(--font-sans); font-size:1.15rem; font-weight:700; color:var(--text-primary); margin:4px 0 12px 0;">
                    {cf.factual_action.replace('_', ' ').upper()}
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                    <div class="metric-cell">
                        <div class="metric-cell-label">PEAK CORE TEMP</div>
                        <div class="metric-cell-value">{cf.factual_t_core:.1f}<span class="metric-cell-unit">°C</span></div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-cell-label">COOLANT FLOW</div>
                        <div class="metric-cell-value">{cf.factual_flow:.1f}<span class="metric-cell-unit">L/m</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cf:
        st.markdown(
            f"""
            <div class="scada-panel" style="border-top:2px solid var(--color-cyan);">
                <div style="font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:var(--color-cyan); text-transform:uppercase; letter-spacing:0.06em;">
                    COUNTERFACTUAL WORLD (INTERVENED)
                </div>
                <div style="font-family:var(--font-sans); font-size:1.15rem; font-weight:700; color:var(--color-cyan); margin:4px 0 12px 0;">
                    {cf.counterfactual_action.replace('_', ' ').upper()}
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                    <div class="metric-cell">
                        <div class="metric-cell-label">COUNTERFACTUAL T_core</div>
                        <div class="metric-cell-value" style="color:#34d399;">{cf.counterfactual_t_core:.1f}<span class="metric-cell-unit">°C</span></div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-cell-label">COUNTERFACTUAL FLOW</div>
                        <div class="metric-cell-value" style="color:#38bdf8;">{cf.counterfactual_flow:.1f}<span class="metric-cell-unit">L/m</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Net Causal Deltas
    st.markdown("##### NET CAUSAL DELTAS: $\Delta = Y(do(A_{\\text{alt}})) - Y(A_{\\text{fact}})$")
    d1, d2, d3 = st.columns(3)
    with d1:
        dt_col = "#34d399" if cf.delta_t_core <= 0 else "#f43f5e"
        dt_desc = "Thermal Cooling Effect" if cf.delta_t_core < 0 else "Heating Penalty"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">Δ CORE TEMPERATURE</div>
                <div class="metric-cell-value" style="color:{dt_col};">{cf.delta_t_core:+.2f}<span class="metric-cell-unit">°C</span></div>
                <div class="metric-cell-footer">
                    <span>{dt_desc.upper()}</span>
                    <span>NET EFFECT</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        df_col = "#38bdf8"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">Δ COOLANT FLOW</div>
                <div class="metric-cell-value" style="color:{df_col};">{cf.delta_flow:+.2f}<span class="metric-cell-unit">L/min</span></div>
                <div class="metric-cell-footer">
                    <span>VOLUMETRIC FLUX</span>
                    <span>CIRCULATION</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d3:
        dp_col = "#38bdf8" if abs(cf.delta_pressure) < 0.8 else "#f59e0b"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">Δ SYSTEM PRESSURE</div>
                <div class="metric-cell-value" style="color:{dp_col};">{cf.delta_pressure:+.2f}<span class="metric-cell-unit">bar</span></div>
                <div class="metric-cell-footer">
                    <span>HYDRODYNAMIC HEAD</span>
                    <span>BACKPRESSURE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
