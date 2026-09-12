"""
Rejected Alternatives & Candidate Matrix Panel Component (Task 7.2 / Phase 7.7).
Renders the multi-objective optimization Pareto candidate matrix.
Strictly zero emojis; high-density SCADA candidate ranking.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st
from prism.explanation.unified_record import PrismDecisionRecord


def render_alternatives_panel(record: PrismDecisionRecord) -> None:
    """Render the candidate evaluation and rejected alternatives matrix."""
    st.markdown(
        """
        <div class="scada-panel-header">
            <div class="scada-panel-title">
                MULTI-OBJECTIVE CANDIDATE EVALUATION & ARBITRATION MATRIX
            </div>
            <div style="font-family:var(--font-mono); font-size:0.70rem; color:var(--text-muted);">
                PARETO OPTIMIZATION // SAFETY FILTERING
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dq = record.decision_quality
    if dq.utility_score is None:
        st.markdown(
            "<div class='scada-panel' style='color:var(--text-muted); font-size:0.85rem;'>"
            "Candidate planning withheld due to upstream Model Distrust Abstention."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Summary Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">OPTIMAL EXPECTED UTILITY</div>
                <div class="metric-cell-value" style="color:var(--color-cyan);">{dq.utility_score:+.4f}</div>
                <div class="metric-cell-footer">
                    <span>RANK #1 CANDIDATE</span>
                    <span>SELECTED</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">SECOND-BEST CANDIDATE</div>
                <div class="metric-cell-value" style="font-size:1.15rem; color:var(--text-secondary);">{dq.second_best_candidate or 'NONE'}</div>
                <div class="metric-cell-footer">
                    <span>RUNNER-UP ALTERNATIVE</span>
                    <span>RANK #2</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        margin_val = f"{dq.decision_margin:+.4f}" if dq.decision_margin is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-cell">
                <div class="metric-cell-label">DECISION MARGIN OVER #2</div>
                <div class="metric-cell-value" style="color:#34d399;">{margin_val}</div>
                <div class="metric-cell-footer">
                    <span>PARETO UTILITY GAP</span>
                    <span>MARGIN</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Rejected Alternatives Table
    st.markdown("##### EVALUATED ACTION CANDIDATE ROSTER")
    if dq.alternatives_rejected:
        rows = []
        for alt in dq.alternatives_rejected:
            cand_id = alt.get("candidate_id", "Unknown")
            u_score = alt.get("utility_score")
            u_str = f"{u_score:+.4f}" if u_score is not None else "N/A"
            is_safe = alt.get("is_safe", False)
            reason = alt.get("rejection_reason", "Suboptimal Utility Score")
            rows.append({
                "Candidate Action": cand_id.replace("cand_", "").upper(),
                "Expected Utility": u_str,
                "Safety Status": "COMPLIANT" if is_safe else "HARD BREACH",
                "Outcome": "REJECTED",
                "Arbitration Rationale": reason,
            })
        df_alt = pd.DataFrame(rows)
        st.dataframe(df_alt, use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='scada-panel' style='color:var(--text-muted); font-size:0.85rem;'>Single admissible candidate or inaction policy available.</div>", unsafe_allow_html=True)
