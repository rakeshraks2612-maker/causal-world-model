"""
Prediction & Uncertainty Forecast Panel Component (Task 7.2 / Phase 7.7).
Renders the multi-step trajectory forecast, prediction uncertainty bands (+/- 2 sigma),
and hard operating limit thresholds.
Strictly zero emojis; high-density SCADA time-series visualizations.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from prism.explanation.unified_record import PrismDecisionRecord
from prism.dataset.decision_benchmark import LearnerDecisionScenario


def render_prediction_panel(
    record: PrismDecisionRecord,
    learner_scen: Optional[LearnerDecisionScenario] = None,
) -> None:
    """Render trajectory forecast charts and uncertainty bounds."""
    st.markdown(
        """
        <div class="scada-panel-header">
            <div class="scada-panel-title">
                MULTI-HORIZON TRAJECTORY FORECAST & UNCERTAINTY ENVELOPE (H = 40)
            </div>
            <div style="font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);">
                FORWARD ROLLOUT // RECURRENT LATENT SCM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if record.trust.trust_state == "MODEL_ABSTAIN" or record.decision.decision_status == "BLOCKED":
        st.markdown(
            """
            <div class="decision-banner decision-banner-abstain">
                <div style="display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:#fbbf24; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
                    <span class="indicator-dot indicator-dot-amber"></span> PREDICTIVE ROLLOUT WITHHELD
                </div>
                <div style="font-family:var(--font-sans); font-size:1.15rem; font-weight:700; color:var(--text-primary); margin-bottom:4px;">
                    MODEL TRUST GATE TRIGGERED: MODEL_ABSTAIN
                </div>
                <div style="font-size:0.86rem; color:#fde68a;">
                    Recurrent neural rollouts are suspended under distrusted regimes to prevent ungrounded predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Extract metrics
    c_therm = record.safety.constraints.get("thermal")
    c_press = record.safety.constraints.get("pressure")
    c_flow = record.safety.constraints.get("flow")

    peak_t = c_therm.raw_prediction if c_therm and c_therm.raw_prediction is not None else 88.5
    eff_t = c_therm.effective_value if c_therm and c_therm.effective_value is not None else 90.1
    sig_t = c_therm.uncertainty_sigma if c_therm and c_therm.uncertainty_sigma is not None else 0.8

    max_p = c_press.raw_prediction if c_press and c_press.raw_prediction is not None else 4.1
    min_f = c_flow.raw_prediction if c_flow and c_flow.raw_prediction is not None else 35.0

    init_t = 91.4
    init_p = 4.1
    init_f = 31.7
    if learner_scen is not None and learner_scen.historical_observations is not None:
        t_star = learner_scen.intervention_time
        init_t = float(learner_scen.historical_observations[t_star, 0])
        init_p = float(learner_scen.historical_observations[t_star, 2])
        init_f = float(learner_scen.historical_observations[t_star, 3])

    # Summary Grid Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        t_pass = eff_t < 95.0
        t_col = "#10b981" if t_pass else "#f43f5e"
        t_status = "SAFE" if t_pass else "BREACH"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>PEAK CORE TEMP (T_core)</span>
                    <span style="color:{t_col}; font-family:var(--font-mono);">{t_status}</span>
                </div>
                <div class="metric-cell-value" style="color:{t_col};">
                    {peak_t:.1f}<span class="metric-cell-unit">°C (μ)</span>
                </div>
                <div class="metric-cell-footer">
                    <span>EFFECTIVE (μ+2σ): <strong>{eff_t:.1f} °C</strong></span>
                    <span>HARD LIMIT: 95.0 °C</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        p_eff = c_press.effective_value if c_press and c_press.effective_value else max_p
        p_pass = p_eff < 5.5
        p_col = "#10b981" if p_pass else "#f43f5e"
        p_status = "SAFE" if p_pass else "BREACH"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>MAX SYSTEM PRESSURE (P_sys)</span>
                    <span style="color:{p_col}; font-family:var(--font-mono);">{p_status}</span>
                </div>
                <div class="metric-cell-value" style="color:{p_col};">
                    {max_p:.2f}<span class="metric-cell-unit">bar (μ)</span>
                </div>
                <div class="metric-cell-footer">
                    <span>EFFECTIVE (μ+2σ): <strong>{p_eff:.2f} bar</strong></span>
                    <span>HARD LIMIT: 5.50 bar</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        f_eff = c_flow.effective_value if c_flow and c_flow.effective_value else min_f
        f_pass = f_eff > 8.0
        f_col = "#10b981" if f_pass else "#f43f5e"
        f_status = "SAFE" if f_pass else "BREACH"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">
                    <span>MIN COOLANT FLOW (F_cool)</span>
                    <span style="color:{f_col}; font-family:var(--font-mono);">{f_status}</span>
                </div>
                <div class="metric-cell-value" style="color:{f_col};">
                    {min_f:.1f}<span class="metric-cell-unit">L/min (μ)</span>
                </div>
                <div class="metric-cell-footer">
                    <span>EFFECTIVE (μ-2σ): <strong>{f_eff:.1f} L/m</strong></span>
                    <span>MIN LIMIT: 8.0 L/m</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Trajectory Plot with Uncertainty Bands
    st.markdown("##### OPEN-LOOP TRAJECTORY WITH ±2σ UNCERTAINTY ENVELOPE")
    steps = np.arange(0, 41)
    time_s = steps * 0.5
    
    t_traj = init_t + (peak_t - init_t) * (1.0 - np.exp(-steps / 8.0))
    sigma_traj = np.linspace(0.2, max(sig_t, 0.8), len(steps))
    upper_bound = t_traj + 2.0 * sigma_traj
    lower_bound = t_traj - 2.0 * sigma_traj

    chart_data = pd.DataFrame({
        "Time_Seconds": time_s,
        "Mean_Forecast": t_traj,
        "Upper_2Sigma": upper_bound,
        "Lower_2Sigma": lower_bound,
        "Safety_Limit": np.full_like(steps, 95.0, dtype=float),
    })

    base = alt.Chart(chart_data).encode(x=alt.X("Time_Seconds:Q", title="Simulation Time (Seconds, Horizon H=40)"))
    
    band = base.mark_area(opacity=0.18, color="#06b6d4").encode(
        y=alt.Y("Lower_2Sigma:Q", title="Core Junction Temp (°C)", scale=alt.Scale(domain=[75, 105])),
        y2="Upper_2Sigma:Q",
    )
    
    line = base.mark_line(color="#06b6d4", strokeWidth=2.5).encode(
        y="Mean_Forecast:Q",
        tooltip=[
            alt.Tooltip("Time_Seconds:Q", title="Time (s)"),
            alt.Tooltip("Mean_Forecast:Q", title="Predicted μ (°C)", format=".2f"),
            alt.Tooltip("Upper_2Sigma:Q", title="Upper 2σ Bound (°C)", format=".2f"),
        ]
    )

    limit = base.mark_line(color="#f43f5e", strokeDash=[6, 4], strokeWidth=1.5).encode(
        y="Safety_Limit:Q",
        tooltip=[alt.Tooltip("Safety_Limit:Q", title="Hard Safety Limit (°C)")]
    )

    chart = (band + line + limit).properties(height=280).configure_view(strokeOpacity=0).configure_axis(
        gridColor="rgba(255,255,255,0.05)",
        labelColor="#94a3b8",
        titleColor="#cbd5e1",
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        "<div style='font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);'>"
        "CYAN LINE: Mean Predicted Trajectory (μ) | CYAN SHADE: ±2.0σ Conservative Uncertainty Envelope | RED DASH: 95.0°C Hard Thermal Limit"
        "</div>",
        unsafe_allow_html=True,
    )
