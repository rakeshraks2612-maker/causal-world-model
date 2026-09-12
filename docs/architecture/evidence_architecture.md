# PRISM Evidence & Auditability Architecture

This document describes PRISM's evidence generation layer, the 8-domain unified decision record (`PrismDecisionRecord`), canonical serialization protocols, and the deterministic SHA-256 cryptographic audit framework.

---

## 1. Architectural Role: Pure Projection Layer

A central invariant of PRISM is that the **Evidence Assembly Layer is a pure immutable projection**.

```text
Decision / Planning Stack (Authoritative)
   ├── Trust Gateway
   ├── Causal Engine
   ├── Counterfactual Simulator
   ├── Safety Constraint Evaluator
   └── Planner / Cost Model
            │
            ▼ (Read-Only Outputs)
┌───────────────────────────────────────┐
│       EVIDENCE ASSEMBLY LAYER         │
│  • Pure deterministic serialization   │
│  • Zero reasoning / Zero decisioning  │
│  • Compiles 8-Domain Evidence Record  │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│     CANONICAL JSON SERIALIZATION      │
│  • Sorted keys (alphabetical)         │
│  • Deterministic float representation │
│  • Whitespace-invariant formatting    │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│      DETERMINISTIC SHA-256 HASH       │
│  • 256-bit cryptographic fingerprint  │
│  • 100% tamper-evident auditability   │
└───────────────────────────────────────┘
```

> [!IMPORTANT]
> **No Reasoning Duplication**: The evidence layer never calculates new safety margins, never re-runs counterfactuals, and never alters planner rankings. It faithfully reflects upstream authoritative records.

---

## 2. The 8 Unified Evidence Domains

Every decision or abstention produces a structured `PrismDecisionRecord` spanning eight standardized domains:

```text
PrismDecisionRecord
├── 1. decision          (Action name, status, commanded value, expected utility)
├── 2. trust             (R_T, R_8D, D_latent, trust boundary status, gating decision)
├── 3. causal_reasoning  (Structural DAG attribution, mechanism path deltas)
├── 4. counterfactual    (Abduced noise, twin-world trajectory, ΔT_core, ΔP_sys, ΔF_cool)
├── 5. safety            (Effective bounds, thermal/hydraulic margins, limiting constraint)
├── 6. abstention        (Root cause diagnostic, excess threshold, blocked actions)
├── 7. decision_quality  (Model confidence, regret estimates, alternate candidate rankings)
└── 8. provenance        (Git commit, pipeline version, configuration hash, SHA-256 hash)
```

### Domain Descriptions
1. **`decision`**: The commanded action, operational intent, ranking among candidate alternatives, and execution status (`RECOMMENDED`, `MARGINAL`, `UNSAFE`, `BLOCKED`).
2. **`trust`**: Upstream model-trust telemetry including thermal residual $R_T$, 8D normalized residual $R_{8D}$, and latent Mahalanobis distance $D_{\text{latent}}$.
3. **`causal_reasoning`**: Qualitative and quantitative path attribution along structural causal graph edges (e.g. $A_{\text{pump}} \to F_{\text{cool}} \to T_{\text{cool}} \to T_{\text{core}}$).
4. **`counterfactual`**: Retrospective twin-world simulation comparing the factual trajectory against the counterfactual intervention under identical exogenous disturbances.
5. **`safety`**: Conservative uncertainty-adjusted peak values ($T_{\text{eff}} = \mu + 2\sigma$), physical constraint headroom margins, and identified limiting constraints.
6. **`abstention`**: Structured diagnostic data emitted if trust boundaries are violated, detailing the primary reason, excess threshold, and blocked actuation paths.
7. **`decision_quality`**: Statistical confidence indicators, oracle regret estimates, and comparative candidate metrics.
8. **`provenance`**: Machine-readable environment metadata including pipeline version, model checkpoint ID, random seed, and cryptographic hash.

---

## 3. Cryptographic Provenance & Tamper-Evident Hashing

### 3.1 Canonical JSON Serialization
To guarantee that two identical decision records always produce the exact same cryptographic hash across different machines, operating systems, and Python runtimes, PRISM applies canonical serialization rules:
1. **Sorted Keys**: All JSON dictionary keys are sorted recursively in strict alphabetical order (`sort_keys=True`).
2. **Whitespace Invariance**: Zero insignificant whitespace, indentation, or trailing newlines (`separators=(',', ':')`).
3. **UTF-8 Encoding**: Character encoding is strictly enforced as UTF-8.

### 3.2 Deterministic SHA-256 Fingerprint
The canonical serialized byte stream is hashed using SHA-256:

$$\text{EvidenceFingerprint} = \text{SHA-256}(\text{CanonicalJSON}(\text{PrismDecisionRecord}))$$

### 3.3 Tamper-Evident Integrity Property
Because SHA-256 is collision-resistant:
* **Fidelity**: Any post-hoc modification to decision values, safety margins, timestamps, or causal path attributions completely changes the resulting 256-bit hash.
* **Auditability**: Operators, regulators, or safety auditors can independently verify whether a stored decision record matches its recorded provenance fingerprint.

> [!NOTE]
> **Correct Terminology**: PRISM provides **deterministic SHA-256 evidence fingerprinting and tamper-evident auditability**. It is not a distributed blockchain or decentralized ledger.
