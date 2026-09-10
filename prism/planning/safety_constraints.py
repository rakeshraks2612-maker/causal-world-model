"""Canonical Hard Safety Constraints, Envelopes, and Result Contract (Task 5.3).

Defines:
1. SafetySeverity: Categorical severity classification (SAFE, MARGINAL, UNSAFE, ABSTAIN_REQUIRED).
2. SafetyViolationReason: Standardized machine-readable violation codes.
3. DecisionSafetyConfig: Canonical single source of truth for hard physical envelopes and buffers.
4. SafetyResult: Rich structured result tracking per-variable safety, margins, violations, and severity.
5. SafetyConstraintEngine: Authoritative safety gate enforcing exact boundary semantics,
   multi-violation detection, NaN/Inf fail-closed behavior, and uncertainty-aware safety envelopes.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SafetySeverity(str, Enum):
    """Categorical classification of candidate safety posture."""
    SAFE = "SAFE"                       # All constraints met with comfortable margins
    MARGINAL = "MARGINAL"               # Within hard envelope, but near boundary buffer
    UNSAFE = "UNSAFE"                   # Hard physical constraint violation predicted
    ABSTAIN_REQUIRED = "ABSTAIN_REQUIRED" # OOD latent state, excessive epistemic variance, or invalid state


class SafetyViolationReason(str, Enum):
    """Standardized machine-readable violation reason codes for explanation and audit layers."""
    THERMAL_LIMIT_EXCEEDED = "THERMAL_LIMIT_EXCEEDED"
    PRESSURE_LIMIT_EXCEEDED = "PRESSURE_LIMIT_EXCEEDED"
    FLOW_MINIMUM_VIOLATED = "FLOW_MINIMUM_VIOLATED"
    LATENT_NOVELTY_EXCEEDED = "LATENT_NOVELTY_EXCEEDED"
    UNCERTAINTY_SAFETY_VIOLATION = "UNCERTAINTY_SAFETY_VIOLATION"
    INVALID_NUMERICAL_STATE = "INVALID_NUMERICAL_STATE"


@dataclass
class DecisionSafetyConfig:
    """Canonical single source of truth for safety constraints and operating envelopes."""

    # 1. Hard Physical & Epistemic Boundaries
    thermal_hard_limit: float = 95.0         # Max permissible core temperature (°C)
    pressure_hard_limit: float = 5.50        # Max permissible system pressure (bar)
    flow_hard_limit: float = 8.00            # Min permissible coolant flow (L/min)
    latent_novelty_hard_limit: float = 15.0  # Max allowable OOD latent Mahalanobis distance

    # 2. Marginal Buffer Zone Thresholds (Warning envelope)
    thermal_marginal_threshold: float = 85.0 # Core temp above 85°C flagged as MARGINAL
    pressure_marginal_threshold: float = 4.80 # Pressure above 4.80 bar flagged as MARGINAL
    flow_marginal_threshold: float = 12.00   # Flow below 12.0 L/min flagged as MARGINAL
    latent_marginal_threshold: float = 10.0  # Latent distance above 10.0 flagged as MARGINAL

    # 3. Uncertainty-Aware Policy Parameters
    uncertainty_multiplier_k: float = 2.0    # Number of standard deviations k (e.g. k=2.0 ~ 95-97.5% confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyResult:
    """Detailed structured assessment of candidate intervention safety."""

    is_safe: bool
    severity: SafetySeverity
    thermal_safe: bool
    pressure_safe: bool
    flow_safe: bool
    support_safe: bool
    violations: List[str]
    violation_reasons: List[str]
    margin_thermal: float                    # 95.0 - T_core (positive = safe buffer)
    margin_pressure: float                   # 5.50 - P_sys (positive = safe buffer)
    margin_flow: float                       # F_cool - 8.00 (positive = safe buffer)
    margin_support: float                    # 15.0 - D_latent (positive = within support)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": bool(self.is_safe),
            "severity": self.severity.value,
            "thermal_safe": bool(self.thermal_safe),
            "pressure_safe": bool(self.pressure_safe),
            "flow_safe": bool(self.flow_safe),
            "support_safe": bool(self.support_safe),
            "violations": list(self.violations),
            "violation_reasons": list(self.violation_reasons),
            "margin_thermal": float(self.margin_thermal),
            "margin_pressure": float(self.margin_pressure),
            "margin_flow": float(self.margin_flow),
            "margin_support": float(self.margin_support),
        }


class SafetyConstraintEngine:
    """Authoritative safety evaluator implementing exact boundary semantics and fail-closed checks."""

    def __init__(self, config: Optional[DecisionSafetyConfig] = None) -> None:
        self.config = config or DecisionSafetyConfig()

    def evaluate(
        self,
        peak_t_core: float,
        max_pressure: float,
        min_flow: float,
        latent_novelty: float = 0.0,
        t_core_std: float = 0.0,
        p_sys_std: float = 0.0,
        f_cool_std: float = 0.0,
        use_uncertainty_bounds: bool = False,
    ) -> SafetyResult:
        """Evaluate candidate safety against hard boundaries and uncertainty envelopes.
        
        Boundary semantics:
        - T_core >= thermal_hard_limit (95.0°C) -> FAIL
        - P_sys >= pressure_hard_limit (5.50 bar) -> FAIL
        - F_cool <= flow_hard_limit (8.00 L/min) -> FAIL
        - Latent novelty > latent_novelty_hard_limit (15.0) -> FAIL
        - NaN / Inf in any input -> FAIL CLOSED (ABSTAIN_REQUIRED)
        """
        # 1. Fail-closed on NaN or Inf in any parameter
        values_to_check = [
            peak_t_core, max_pressure, min_flow, latent_novelty,
            t_core_std, p_sys_std, f_cool_std
        ]
        if any(np.isnan(v) or np.isinf(v) for v in values_to_check):
            return SafetyResult(
                is_safe=False,
                severity=SafetySeverity.ABSTAIN_REQUIRED,
                thermal_safe=False,
                pressure_safe=False,
                flow_safe=False,
                support_safe=False,
                violations=["Invalid numerical state detected (NaN or Inf encountered in prediction/uncertainty)."],
                violation_reasons=[SafetyViolationReason.INVALID_NUMERICAL_STATE.value],
                margin_thermal=-float("inf"),
                margin_pressure=-float("inf"),
                margin_flow=-float("inf"),
                margin_support=-float("inf"),
            )

        # 2. Compute effective values (point estimate vs uncertainty-adjusted upper/lower bounds)
        k = self.config.uncertainty_multiplier_k if use_uncertainty_bounds else 0.0
        eff_t_core = float(peak_t_core + k * max(0.0, t_core_std))
        eff_pressure = float(max_pressure + k * max(0.0, p_sys_std))
        eff_flow = float(min_flow - k * max(0.0, f_cool_std))
        eff_novelty = float(latent_novelty)

        # 3. Margins
        margin_thermal = float(self.config.thermal_hard_limit - eff_t_core)
        margin_pressure = float(self.config.pressure_hard_limit - eff_pressure)
        margin_flow = float(eff_flow - self.config.flow_hard_limit)
        margin_support = float(self.config.latent_novelty_hard_limit - eff_novelty)

        # 4. Evaluate individual physical & support constraints
        thermal_safe = (eff_t_core < self.config.thermal_hard_limit)
        pressure_safe = (eff_pressure < self.config.pressure_hard_limit)
        flow_safe = (eff_flow > self.config.flow_hard_limit)
        support_safe = (eff_novelty <= self.config.latent_novelty_hard_limit)

        violations: List[str] = []
        violation_reasons: List[str] = []

        if not thermal_safe:
            unc_tag = " (uncertainty-adjusted)" if use_uncertainty_bounds and t_core_std > 0 else ""
            violations.append(
                f"Peak core temperature{unc_tag} ({eff_t_core:.2f}°C) exceeds hard threshold ({self.config.thermal_hard_limit:.2f}°C)."
            )
            violation_reasons.append(
                SafetyViolationReason.UNCERTAINTY_SAFETY_VIOLATION.value
                if (use_uncertainty_bounds and peak_t_core < self.config.thermal_hard_limit)
                else SafetyViolationReason.THERMAL_LIMIT_EXCEEDED.value
            )

        if not pressure_safe:
            unc_tag = " (uncertainty-adjusted)" if use_uncertainty_bounds and p_sys_std > 0 else ""
            violations.append(
                f"Peak system pressure{unc_tag} ({eff_pressure:.2f} bar) exceeds hard limit ({self.config.pressure_hard_limit:.2f} bar)."
            )
            violation_reasons.append(
                SafetyViolationReason.UNCERTAINTY_SAFETY_VIOLATION.value
                if (use_uncertainty_bounds and max_pressure < self.config.pressure_hard_limit)
                else SafetyViolationReason.PRESSURE_LIMIT_EXCEEDED.value
            )

        if not flow_safe:
            unc_tag = " (uncertainty-adjusted)" if use_uncertainty_bounds and f_cool_std > 0 else ""
            violations.append(
                f"Minimum coolant flow{unc_tag} ({eff_flow:.2f} L/min) below safety threshold ({self.config.flow_hard_limit:.2f} L/min)."
            )
            violation_reasons.append(
                SafetyViolationReason.UNCERTAINTY_SAFETY_VIOLATION.value
                if (use_uncertainty_bounds and min_flow > self.config.flow_hard_limit)
                else SafetyViolationReason.FLOW_MINIMUM_VIOLATED.value
            )

        if not support_safe:
            violations.append(
                f"Latent manifold novelty ({eff_novelty:.2f}) exceeds epistemic support limit ({self.config.latent_novelty_hard_limit:.2f})."
            )
            violation_reasons.append(SafetyViolationReason.LATENT_NOVELTY_EXCEEDED.value)

        # 5. Determine Overall Safety & Severity Classification
        all_passed = thermal_safe and pressure_safe and flow_safe and support_safe

        if not support_safe:
            severity = SafetySeverity.ABSTAIN_REQUIRED
            is_safe = False
        elif not (thermal_safe and pressure_safe and flow_safe):
            severity = SafetySeverity.UNSAFE
            is_safe = False
        else:
            is_safe = True
            # Check for marginal warning conditions
            is_marginal = (
                eff_t_core >= self.config.thermal_marginal_threshold
                or eff_pressure >= self.config.pressure_marginal_threshold
                or eff_flow <= self.config.flow_marginal_threshold
                or eff_novelty >= self.config.latent_marginal_threshold
            )
            severity = SafetySeverity.MARGINAL if is_marginal else SafetySeverity.SAFE

        return SafetyResult(
            is_safe=is_safe,
            severity=severity,
            thermal_safe=thermal_safe,
            pressure_safe=pressure_safe,
            flow_safe=flow_safe,
            support_safe=support_safe,
            violations=violations,
            violation_reasons=violation_reasons,
            margin_thermal=margin_thermal,
            margin_pressure=margin_pressure,
            margin_flow=margin_flow,
            margin_support=margin_support,
        )
