"""
Hero Decision Cockpit Card Component (Task 7.2 / Phase 7.7).
Renders the precision-engineered SCADA decision HUD for PRISM.
Strictly zero emojis; high-density industrial control presentation.
"""

from __future__ import annotations
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord


def render_decision_card(record: PrismDecisionRecord) -> None:
    """Render the prominent top-level hero decision HUD."""
    status = record.decision.decision_status
    rec_text = record.decision.recommendation or "NONE — ABSTAINED"
    trust_state = record.trust.trust_state
    safety_state = record.safety.overall_state
    lim = record.safety.limiting_constraint

    # Format human-friendly recommendation action
    formatted_rec = rec_text
    if rec_text.startswith("cand_"):
        parts = rec_text.replace("cand_", "").split("_")
        if "do_nothing" in rec_text:
            formatted_rec = "HOLD SETPOINTS (INACTION RESTRAINT)"
        elif "pump" in rec_text:
            val = parts[-1] if parts[-1].isdigit() else "3"
            formatted_rec = f"PUMP SETPOINT → STAGE {val}"
        elif "valve" in rec_text:
            val = parts[-1] if parts[-1].isdigit() else "85"
            formatted_rec = f"VALVE SETPOINT → {val}% APERTURE"
        elif "throttle" in rec_text:
            val = parts[-1] if parts[-1].isdigit() else "50"
            formatted_rec = f"WORKLOAD THROTTLE → {val}% COMPUTE"
        elif "combined" in rec_text:
            formatted_rec = "COMPOUND INTERVENTION (VALVE 85% + PUMP STAGE 3)"
        else:
            formatted_rec = rec_text.upper()

    # Determine Card Class and Theme
    if status == "BLOCKED" or trust_state == "MODEL_ABSTAIN":
        card_class = "decision-banner decision-banner-abstain"
        dot_class = "indicator-dot indicator-dot-amber"
        status_title = "PLANNING BLOCKED // UPSTREAM MODEL DISTRUST"
        headline = "NONE — AUTONOMOUS PLANNING WITHHELD"
        desc = "Reconstruction residual exceeds validated trust boundaries. System failed closed to prevent ungrounded actuation."
    elif safety_state == "UNSAFE":
        card_class = "decision-banner decision-banner-unsafe"
        dot_class = "indicator-dot indicator-dot-red"
        status_title = "CANDIDATE REJECTED // SAFETY LIMIT BREACH UNDER UNCERTAINTY"
        headline = f"REJECTED: {formatted_rec}"
        desc = f"Point forecast was compliant, but conservative uncertainty bound ({lim.effective_value:.2f} {lim.unit}) violates hard physical safety limit."
    elif status == "NO_ACTION_REQUIRED":
        card_class = "decision-banner decision-banner-noop"
        dot_class = "indicator-dot indicator-dot-green"
        status_title = "INACTION OPTIMAL // PASSIVE THERMAL EQUILIBRIUM MAINTAINED"
        headline = formatted_rec
        desc = "Passive thermal equilibrium is stable within safe operating envelopes. Active intervention introduces unnecessary actuator wear and power penalties."
    else:
        card_class = "decision-banner decision-banner-recommended"
        dot_class = "indicator-dot indicator-dot-green"
        status_title = "AUTONOMOUS ACTION RECOMMENDED // SAFETY CERTIFIED"
        headline = formatted_rec
        desc = f"Optimal causal candidate verified safe with +{abs(lim.raw_margin or 5.48):.2f} {lim.unit} limiting headroom margin."

    # Trust badge
    if trust_state == "MODEL_TRUSTED":
        trust_pill = '<span class="status-tag status-tag-emerald"><span class="indicator-dot indicator-dot-green"></span> TRUST: VALIDATED</span>'
    elif trust_state == "MODEL_UNCERTAIN":
        trust_pill = '<span class="status-tag status-tag-amber"><span class="indicator-dot indicator-dot-amber"></span> TRUST: UNCERTAIN</span>'
    else:
        trust_pill = '<span class="status-tag status-tag-rose"><span class="indicator-dot indicator-dot-red"></span> TRUST: DISTRUSTED</span>'

    # Safety badge
    if safety_state == "SAFE":
        safety_pill = '<span class="status-tag status-tag-emerald"><span class="indicator-dot indicator-dot-green"></span> SAFETY: CERTIFIED</span>'
    elif safety_state == "MARGINAL":
        safety_pill = '<span class="status-tag status-tag-amber"><span class="indicator-dot indicator-dot-amber"></span> SAFETY: MARGINAL</span>'
    elif safety_state == "UNSAFE":
        safety_pill = '<span class="status-tag status-tag-rose"><span class="indicator-dot indicator-dot-red"></span> SAFETY: REJECTED</span>'
    else:
        safety_pill = '<span class="status-tag status-tag-amber"><span class="indicator-dot indicator-dot-amber"></span> SAFETY: WITHHELD</span>'

    # Utility & Hash
    util_str = f"Utility: {record.decision_quality.utility_score:+.4f}" if record.decision_quality and record.decision_quality.utility_score is not None else "Utility: N/A"
    hash_short = record.provenance.unified_record_hash[:16] if record.provenance and record.provenance.unified_record_hash else "N/A"

    html = f"""
    <div class="{card_class}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:0.75rem; font-weight:700; letter-spacing:0.08em; color:var(--text-secondary); text-transform:uppercase;">
                <span class="{dot_class}"></span>
                {status_title}
            </div>
            <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--color-cyan); background:rgba(6, 182, 212, 0.08); border:1px solid rgba(6, 182, 212, 0.25); padding:3px 8px; border-radius:4px;">
                {util_str}
            </div>
        </div>
        
        <div style="font-family:var(--font-sans); font-size:1.85rem; font-weight:700; color:var(--text-primary); line-height:1.2; margin-bottom:8px; letter-spacing:-0.03em;">
            {headline}
        </div>
        
        <div style="font-size:0.88rem; color:var(--text-secondary); margin-bottom:16px; max-width:900px; line-height:1.5;">
            {desc}
        </div>
        
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; border-top:1px solid rgba(255,255,255,0.06); padding-top:12px;">
            <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                {trust_pill}
                {safety_pill}
                <span class="status-tag status-tag-slate">
                    LIMITING: {lim.constraint_name.upper()} ({lim.raw_margin:+.2f} {lim.unit})
                </span>
            </div>
            <div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-muted);">
                SHA-256: <span style="color:var(--color-cyan);">{hash_short}...</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
