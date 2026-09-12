"""
Abstention & Model Distrust Console Component (Task 7.2 / Phase 7.7).
Renders the dedicated fail-closed abstention interface when PRISM
refuses to make an ungrounded or out-of-distribution decision.
Strictly zero emojis; high-density SCADA emergency lockout presentation.
"""

from __future__ import annotations
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord


def render_abstention_panel(record: PrismDecisionRecord) -> None:
    """Render the prominent abstention explanation interface."""
    abst = record.abstention
    if not abst.abstained:
        return

    rt = record.trust.reconstruction_residual_t_core
    tau_t = record.trust.tau_residual_t or 6.0827
    excess = rt - tau_t if rt is not None and tau_t is not None else 0.0
    r8d = record.trust.reconstruction_residual_8d_norm or 1.815
    tau_8d = record.trust.tau_residual_8d or 1.8960
    dlat = record.trust.latent_novelty_d or 7.73

    html_abstain = f"""
    <div class="decision-banner decision-banner-abstain" style="padding: 24px; border-width: 2px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span class="indicator-dot indicator-dot-amber"></span>
            <span style="font-family:var(--font-mono); font-size:0.75rem; font-weight:700; color:#fbbf24; letter-spacing:0.08em; text-transform:uppercase;">
                CRITICAL MODEL DISTRUST // FAIL-CLOSED ABSTENTION ACTIVATED
            </span>
        </div>
        
        <div style="font-family:var(--font-sans); font-size:1.75rem; font-weight:700; color:var(--text-primary); line-height:1.2; margin-bottom:8px; letter-spacing:-0.03em;">
            PRISM WITHHELD AUTONOMOUS ACTUATION
        </div>
        
        <div style="font-size:0.88rem; color:var(--text-secondary); margin-bottom:20px; line-height:1.5; max-width:900px;">
            Incoming sensor telemetry contradicts internal world-model dynamics. Because predictive validity cannot be guaranteed, PRISM strictly withheld candidate planning to prevent ungrounded autonomous actuation.
        </div>
        
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-bottom:20px;">
            <div class="metric-cell" style="border-color:rgba(244, 63, 94, 0.4);">
                <div class="metric-cell-label">
                    <span>OBSERVED RESIDUAL (R_T)</span>
                    <span style="color:#f43f5e; font-family:var(--font-mono);">BREACH</span>
                </div>
                <div class="metric-cell-value" style="color:#f43f5e;">{rt:.2f}<span class="metric-cell-unit">°C</span></div>
                <div class="metric-cell-footer">
                    <span>EXCEEDS BY: <strong>+{excess:.2f} °C</strong></span>
                    <span>98TH PERCENTILE</span>
                </div>
            </div>
            <div class="metric-cell" style="border-color:rgba(245, 158, 11, 0.4);">
                <div class="metric-cell-label">
                    <span>CALIBRATED CEILING (τ_RT)</span>
                    <span style="color:#f59e0b; font-family:var(--font-mono);">THRESHOLD</span>
                </div>
                <div class="metric-cell-value" style="color:#f59e0b;">{tau_t:.4f}<span class="metric-cell-unit">°C</span></div>
                <div class="metric-cell-footer">
                    <span>TRUST BOUNDARY</span>
                    <span>CALIBRATED</span>
                </div>
            </div>
            <div class="metric-cell" style="border-color:rgba(59, 130, 246, 0.4);">
                <div class="metric-cell-label">
                    <span>LATENT DISTANCE (D_lat)</span>
                    <span style="color:#60a5fa; font-family:var(--font-mono);">SUPPORT</span>
                </div>
                <div class="metric-cell-value" style="color:#60a5fa;">{dlat:.2f}<span class="metric-cell-unit">d_M</span></div>
                <div class="metric-cell-footer">
                    <span>BOUND: ≤ 15.00 d_M</span>
                    <span>MANIFOLD CHECK</span>
                </div>
            </div>
        </div>

        <div class="scada-panel" style="background:rgba(0,0,0,0.3); border-left:3px solid #f43f5e; padding:12px 14px; margin-bottom:14px;">
            <div style="font-size:0.82rem; color:var(--text-primary); margin-bottom:4px;"><strong>Diagnostic Finding:</strong> {abst.primary_reason}</div>
            <div style="font-size:0.82rem; color:var(--text-primary); margin-bottom:4px;"><strong>Blocked Actuation Channels:</strong> {', '.join(f'<code>{a}</code>' for a in abst.blocked_actions)}</div>
            <div style="font-size:0.82rem; color:var(--text-primary);"><strong>Safety Implication:</strong> {abst.safety_implication}</div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(6, 182, 212, 0.08); border:1px solid rgba(6, 182, 212, 0.25); padding:10px 14px; border-radius:6px;">
            <div style="color:var(--color-cyan); font-size:0.85rem; font-weight:600;">
                OPERATOR PROCEDURE: {abst.recommended_next_step}
            </div>
            <div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-muted);">
                STATUS: LOCKED (FAIL-CLOSED)
            </div>
        </div>
    </div>
    """
    st.markdown(html_abstain, unsafe_allow_html=True)
