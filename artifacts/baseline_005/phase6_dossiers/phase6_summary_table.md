# 📊 PRISM Phase 6 Benchmark Evaluation & Evidence Dossier Summary

| Scenario | Title | Recommendation | Trust State | Safety State | Limiting Constraint | Margin | Utility | SHA-256 Fingerprint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `scenario_01_do_nothing` | **Scenario 1: Inaction Restraint** | `cand_do_nothing` | `MODEL_TRUSTED` | `MARGINAL` | `thermal` | `+4.92 °C` | `-0.0218` | `1f0f245bf1fb...` |
| `scenario_02_valve` | **Scenario 2: Valve Intervention** | `cand_pump_4` | `MODEL_TRUSTED` | `UNSAFE` | `thermal` | `-2.16 °C` | `-0.7752` | `5cb725c1af3b...` |
| `scenario_03_throttle` | **Scenario 3: Workload Throttling** | `cand_throttle_50` | `MODEL_TRUSTED` | `MARGINAL` | `thermal` | `+9.83 °C` | `0.1659` | `65762323a77d...` |
| `scenario_04_pump` | **Scenario 4: Pump Head Modulating** | `cand_pump_3` | `MODEL_TRUSTED` | `MARGINAL` | `thermal` | `+5.48 °C` | `0.0034` | `436957cc7fbd...` |
| `scenario_05_combined` | **Scenario 5: Multi-Variable Compound** | `cand_pump_4_only` | `MODEL_TRUSTED` | `MARGINAL` | `thermal` | `+0.57 °C` | `-0.4950` | `6eeb0d61461e...` |
| `scenario_06_all_unsafe` | **Scenario 6: All-Unsafe Abstention** | `NONE` | `MODEL_ABSTAIN` | `ABSTAIN_REQUIRED` | `model_trust_gateway` | `N/A` | `N/A` | `4c62ef5b9fb5...` |

> [!IMPORTANT]
> **Tamper-Evident Integrity:** PRISM produces a deterministic SHA-256 evidence fingerprint that makes post-hoc modification detectable when compared against the recorded provenance hash.