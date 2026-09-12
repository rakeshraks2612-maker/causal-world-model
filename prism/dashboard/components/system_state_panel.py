"""
System State & Telemetry Panel Component (Task 7.2 / Phase 7.7).
Renders high-density industrial SCADA telemetry tiles.
Strictly zero emojis; high-precision engineering symbols.
"""

from __future__ import annotations
from typing import Optional
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord
from prism.dataset.decision_benchmark import LearnerDecisionScenario


def render_system_state_panel(
    record: PrismDecisionRecord,
    learner_scen: Optional[LearnerDecisionScenario] = None,
) -> None:
    """Render the observable telemetry cockpit."""
    st.markdown(
        """
        <div class="scada-panel-header">
            <div class="scada-panel-title">
                PHYSICAL TELEMETRY & LATENT MANIFOLD STATE
            </div>
            <div style="font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);">
                SAMPLE RATE: 2.0 Hz | HORIZON: t=0..40
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Extract historical observation at t*
    if learner_scen is not None and learner_scen.historical_observations is not None:
        t_star = learner_scen.intervention_time
        obs = learner_scen.historical_observations[t_star]
        acts = learner_scen.historical_actions[t_star]
        t_core = float(obs[0])
        t_cool = float(obs[1])
        p_sys = float(obs[2])
        f_cool = float(obs[3])
        l_cpu = float(obs[4])
        v_pos = float(acts[0] * 100.0 if acts[0] <= 1.0 else acts[0])
        p_speed = float(acts[1])
        p_elec = float(obs[7]) if len(obs) > 7 else 4.2
        vib = float(obs[6]) if len(obs) > 6 else 0.8
    else:
        t_core = 91.4
        t_cool = 68.2
        p_sys = 4.1
        f_cool = 31.7
        l_cpu = 87.0
        v_pos = 62.0
        p_speed = 2.0
        p_elec = 4.7
        vib = 0.8

    # Row 1: Primary Thermal-Hydraulics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t_status = "ELEVATED" if t_core >= 88.0 else "NOMINAL"
        t_color = "#f43f5e" if t_core >= 92.0 else ("#f59e0b" if t_core >= 88.0 else "#10b981")
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>CORE TEMP (T_core)</span>
                    <span style="color:{t_color}; font-family:var(--font-mono);">{t_status}</span>
                </div>
                <div class="metric-cell-value" style="color:{t_color};">
                    {t_core:.1f}<span class="metric-cell-unit">°C</span>
                </div>
                <div class="metric-cell-footer">
                    <span>HARD LIMIT: &lt; 95.0 °C</span>
                    <span>MARGIN: {95.0 - t_core:+.1f} °C</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        p_color = "#f43f5e" if p_sys >= 5.0 else "#38bdf8"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>SYSTEM PRESSURE (P_sys)</span>
                    <span style="color:#38bdf8; font-family:var(--font-mono);">STABLE</span>
                </div>
                <div class="metric-cell-value" style="color:{p_color};">
                    {p_sys:.2f}<span class="metric-cell-unit">bar</span>
                </div>
                <div class="metric-cell-footer">
                    <span>HARD LIMIT: &lt; 5.50 bar</span>
                    <span>MARGIN: {5.50 - p_sys:+.2f} bar</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        f_color = "#f43f5e" if f_cool <= 12.0 else "#38bdf8"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>COOLANT FLOW (F_cool)</span>
                    <span style="color:#38bdf8; font-family:var(--font-mono);">CIRCULATING</span>
                </div>
                <div class="metric-cell-value" style="color:{f_color};">
                    {f_cool:.1f}<span class="metric-cell-unit">L/min</span>
                </div>
                <div class="metric-cell-footer">
                    <span>MIN LIMIT: &gt; 8.0 L/min</span>
                    <span>MARGIN: {f_cool - 8.0:+.1f} L/m</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>COOLANT RETURN (T_cool)</span>
                    <span style="color:#34d399; font-family:var(--font-mono);">ACTIVE</span>
                </div>
                <div class="metric-cell-value">
                    {t_cool:.1f}<span class="metric-cell-unit">°C</span>
                </div>
                <div class="metric-cell-footer">
                    <span>HEAT SINK RETURN</span>
                    <span>NOMINAL</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Row 2: Actuators, Workload & Latent Support
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>COMPUTE WORKLOAD (L_cpu)</span>
                    <span style="color:var(--text-muted); font-family:var(--font-mono);">{p_elec:.2f} kW</span>
                </div>
                <div class="metric-cell-value">
                    {l_cpu:.1f}<span class="metric-cell-unit">%</span>
                </div>
                <div class="metric-cell-footer">
                    <span>DEMAND POWER LOAD</span>
                    <span>THERMAL FLUX</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>VALVE POSITION (V_pos)</span>
                    <span style="color:var(--text-muted); font-family:var(--font-mono);">ACTUATOR</span>
                </div>
                <div class="metric-cell-value">
                    {v_pos:.0f}<span class="metric-cell-unit">%</span>
                </div>
                <div class="metric-cell-footer">
                    <span>LINEAR FLOW RESTRICTOR</span>
                    <span>APERTURE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>PUMP SPEED (STAGE)</span>
                    <span style="color:var(--text-muted); font-family:var(--font-mono);">{vib:.2f} mm/s</span>
                </div>
                <div class="metric-cell-value">
                    STAGE {p_speed:.0f}
                </div>
                <div class="metric-cell-footer">
                    <span>HYDRAULIC HEAD DRIVER</span>
                    <span>ROTATION</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c8:
        d_lat = record.trust.latent_novelty_d
        lat_pass = d_lat <= 15.0
        lat_col = "#10b981" if lat_pass else "#f43f5e"
        lat_status = "ON MANIFOLD" if lat_pass else "OOD NOVELTY"
        st.markdown(
            f"""
            <div class="metric-cell" style="border-color:rgba(59, 130, 246, 0.3);">
                <div class="metric-cell-label">
                    <span>LATENT DISTANCE (D_lat)</span>
                    <span style="color:{lat_col}; font-family:var(--font-mono);">{lat_status}</span>
                </div>
                <div class="metric-cell-value" style="color:{lat_col};">
                    {d_lat:.2f}<span class="metric-cell-unit">d_M</span>
                </div>
                <div class="metric-cell-footer">
                    <span>MAHALANOBIS BOUND: ≤ 15.0</span>
                    <span>MARGIN: {15.0 - d_lat:+.2f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
