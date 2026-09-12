"""
Physical Safety & Support Constraints Panel Component (Task 7.2 / Phase 7.7).
Renders the authoritative uncertainty-aware (k=2.0) safety evaluations.
Strictly zero emojis; high-density SCADA headroom metrics and boundary meters.
"""

from __future__ import annotations
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord


def render_safety_panel(record: PrismDecisionRecord) -> None:
    """Render physical safety and model support constraint cards."""
    st.markdown(
        """
        <div class="scada-panel-header">
            <div class="scada-panel-title">
                SAFETY HEADROOM CERTIFICATION // CONSERVATIVE GATING (k = 2.0σ)
            </div>
            <div style="font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);">
                μ ± 2.0σ LOWER/UPPER BOUNDS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lim = record.safety.limiting_constraint
    lim_margin_str = f"{lim.raw_margin:+.2f} {lim.unit}" if lim.raw_margin is not None else "N/A"
    lim_hd_str = f"({lim.normalized_headroom_pct:+.1f}% headroom)" if lim.normalized_headroom_pct is not None else ""

    # Limiting Constraint Callout Card
    if not record.safety.is_safe:
        st.markdown(
            f"""
            <div class="decision-banner decision-banner-unsafe">
                <div style="display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:#fb7185; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
                    <span class="indicator-dot indicator-dot-red"></span> LIMITING SAFETY CONSTRAINT VIOLATED
                </div>
                <div style="font-family:var(--font-sans); font-size:1.25rem; font-weight:700; color:var(--text-primary); margin-bottom:4px;">
                    {lim.constraint_name.upper()} LIMIT BREACHED BY {abs(lim.raw_margin or 0.0):.2f} {lim.unit}
                </div>
                <div style="font-size:0.86rem; color:#fecaca;">
                    {lim.rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="decision-banner decision-banner-recommended">
                <div style="display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:#34d399; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
                    <span class="indicator-dot indicator-dot-green"></span> CERTIFIED SAFE HEADROOM BUFFER
                </div>
                <div style="font-family:var(--font-sans); font-size:1.25rem; font-weight:700; color:var(--text-primary); margin-bottom:4px;">
                    {lim.constraint_name.upper()} MARGIN: {lim_margin_str} {lim_hd_str}
                </div>
                <div style="font-size:0.86rem; color:#a7f3d0;">
                    {lim.rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##### 1. PHYSICAL SAFETY CONSTRAINTS (HARD LIMITS)")
    c_therm = record.safety.constraints.get("thermal")
    c_press = record.safety.constraints.get("pressure")
    c_flow = record.safety.constraints.get("flow")

    col1, col2, col3 = st.columns(3)

    with col1:
        if c_therm and c_therm.effective_value is not None:
            pct = max(0.0, min(100.0, c_therm.normalized_headroom_pct or 0.0))
            t_pass = c_therm.is_passed
            t_col = "#10b981" if t_pass else "#f43f5e"
            t_status = "SAFE" if t_pass else "BREACH"
            st.markdown(
                f"""
                <div class="metric-cell">
                    <div class="metric-cell-label">
                        <span>THERMAL HEADROOM (T_core)</span>
                        <span style="color:{t_col}; font-family:var(--font-mono);">{t_status}</span>
                    </div>
                    <div class="metric-cell-value" style="color:{t_col};">
                        {c_therm.effective_value:.2f}<span class="metric-cell-unit">°C</span>
                    </div>
                    <div class="metric-cell-footer">
                        <span>LIMIT: &lt; 95.0 °C</span>
                        <span>MARGIN: <strong>{c_therm.margin:+.2f} °C</strong></span>
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill" style="background:{t_col}; width:{pct}%;"></div>
                    </div>
                    <div style="font-family:var(--font-mono); font-size:0.68rem; color:var(--text-muted); margin-top:4px;">
                        μ: {c_therm.raw_prediction or 0.0:.2f}°C | σ: {c_therm.uncertainty_sigma or 0.0:.2f}°C (k=2.0)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='metric-cell' style='color:var(--text-muted);'>Evaluation Withheld</div>", unsafe_allow_html=True)

    with col2:
        if c_press and c_press.effective_value is not None:
            pct = max(0.0, min(100.0, c_press.normalized_headroom_pct or 0.0))
            p_pass = c_press.is_passed
            p_col = "#10b981" if p_pass else "#f43f5e"
            p_status = "SAFE" if p_pass else "BREACH"
            st.markdown(
                f"""
                <div class="metric-cell">
                    <div class="metric-cell-label">
                        <span>PRESSURE HEADROOM (P_sys)</span>
                        <span style="color:{p_col}; font-family:var(--font-mono);">{p_status}</span>
                    </div>
                    <div class="metric-cell-value" style="color:{p_col};">
                        {c_press.effective_value:.2f}<span class="metric-cell-unit">bar</span>
                    </div>
                    <div class="metric-cell-footer">
                        <span>LIMIT: &lt; 5.50 bar</span>
                        <span>MARGIN: <strong>{c_press.margin:+.2f} bar</strong></span>
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill" style="background:{p_col}; width:{pct}%;"></div>
                    </div>
                    <div style="font-family:var(--font-mono); font-size:0.68rem; color:var(--text-muted); margin-top:4px;">
                        μ: {c_press.raw_prediction or 0.0:.2f} bar | σ: {c_press.uncertainty_sigma or 0.0:.2f} bar (k=2.0)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='metric-cell' style='color:var(--text-muted);'>Evaluation Withheld</div>", unsafe_allow_html=True)

    with col3:
        if c_flow and c_flow.effective_value is not None:
            pct = max(0.0, min(100.0, c_flow.normalized_headroom_pct or 0.0))
            f_pass = c_flow.is_passed
            f_col = "#10b981" if f_pass else "#f43f5e"
            f_status = "SAFE" if f_pass else "BREACH"
            st.markdown(
                f"""
                <div class="metric-cell">
                    <div class="metric-cell-label">
                        <span>FLOW RATE HEADROOM (F_cool)</span>
                        <span style="color:{f_col}; font-family:var(--font-mono);">{f_status}</span>
                    </div>
                    <div class="metric-cell-value" style="color:{f_col};">
                        {c_flow.effective_value:.2f}<span class="metric-cell-unit">L/m</span>
                    </div>
                    <div class="metric-cell-footer">
                        <span>LIMIT: &gt; 8.0 L/min</span>
                        <span>MARGIN: <strong>{c_flow.margin:+.2f} L/m</strong></span>
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill" style="background:{f_col}; width:{pct}%;"></div>
                    </div>
                    <div style="font-family:var(--font-mono); font-size:0.68rem; color:var(--text-muted); margin-top:4px;">
                        μ: {c_flow.raw_prediction or 0.0:.2f} L/m | σ: {c_flow.uncertainty_sigma or 0.0:.2f} L/m (k=2.0)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='metric-cell' style='color:var(--text-muted);'>Evaluation Withheld</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 2. MODEL LATENT MANIFOLD SUPPORT BOUNDARY")
    c_lat = record.safety.constraints.get("latent_support")
    if c_lat and c_lat.effective_value is not None:
        pct_lat = max(0.0, min(100.0, c_lat.normalized_headroom_pct or 0.0))
        lat_pass = c_lat.is_passed
        lat_col = "#3b82f6" if lat_pass else "#f43f5e"
        lat_tag = "status-tag-emerald" if lat_pass else "status-tag-rose"
        lat_status = "ON MANIFOLD" if lat_pass else "OOD NOVELTY"
        st.markdown(
            f"""
            <div class="metric-cell" style="border-color:rgba(59, 130, 246, 0.3);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="metric-cell-label">LATENT MANIFOLD DISTANCE (D_latent)</div>
                        <div class="metric-cell-value" style="color:{lat_col};">
                            {c_lat.effective_value:.2f}<span class="metric-cell-unit">d_M</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="status-tag {lat_tag}">{lat_status}</span>
                        <div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-muted); margin-top:4px;">
                            MARGIN: {c_lat.margin:+.2f} d_M (BOUND: ≤ 15.0 d_M)
                        </div>
                    </div>
                </div>
                <div class="meter-container">
                    <div class="meter-fill" style="background:{lat_col}; width:{pct_lat}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
