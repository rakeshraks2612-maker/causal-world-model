# PRISM Evidence Assembly Flow Diagram

This document details the aggregation, canonical serialization, and cryptographic fingerprinting of the 8-domain PRISM evidence stack.

---

## 1. ASCII Evidence Flow

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 AUTHORITATIVE ENGINE SUBSYSTEM OUTPUTS                 │
  │                                                                        │
  │  [Planner] ──────► 1. Decision Domain (Action, ranking, utility)       │
  │  [Trust Gate] ───► 2. Trust Domain (R_T, R_8D, D_latent, status)       │
  │  [Causal DAG] ───► 3. Causal Reasoning Domain (Paths, direct effects)  │
  │  [CF Engine] ────► 4. Counterfactual Domain (Twin-world replay, deltas)│
  │  [Safety Gate] ──► 5. Safety Domain (Conservative margins, limits)     │
  │  [Abstention] ───► 6. Abstention Domain (Root cause, blocked actions)  │
  │  [Cost Model] ───► 7. Decision Quality (Confidence, regret estimates)  │
  │  [Environment] ──► 8. Provenance Domain (Version, commit, seed, config)│
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                    PURE IMMUTABLE PROJECTION LAYER                     │
  │  • Zero reasoning / Zero calculation duplication                       │
  │  • Compiles structured PrismDecisionRecord dataclass                   │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                      CANONICAL JSON SERIALIZATION                      │
  │  • Alphabetically sorted keys                                          │
  │  • Whitespace-invariant byte stream                                    │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                       SHA-256 CRYPTOGRAPHIC HASH                       │
  │  • 256-bit tamper-evident fingerprint                                  │
  │  • Modifying any evidence value changes the output hash                │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mermaid Evidence Flow

```mermaid
flowchart TD
    subgraph Upstream["Authoritative Execution Stack (Read-Only)"]
        D1["1. Decision Domain<br/>(Action, status, rank)"]
        D2["2. Trust Domain<br/>(R_T, R_8D, d_M)"]
        D3["3. Causal Domain<br/>(DAG paths, deltas)"]
        D4["4. Counterfactual Domain<br/>(Twin-world trajectory)"]
        D5["5. Safety Domain<br/>(Margins, limiting bounds)"]
        D6["6. Abstention Domain<br/>(Diagnostics, blocks)"]
        D7["7. Quality Domain<br/>(Confidence, regret)"]
        D8["8. Provenance Domain<br/>(Version, seed, config)"]
    end

    subgraph Assembly["Pure Immutable Projection"]
        D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8 --> Record["PrismDecisionRecord<br/>(Immutable Dataclass)"]
    end

    subgraph Crypto["Cryptographic Serialization"]
        Record --> Canon["Canonical JSON Serialization<br/>(Sorted Keys, No Whitespace)"]
        Canon --> SHA["SHA-256 Hash Function"]
        SHA --> Hash["Deterministic Evidence Fingerprint<br/>(Tamper-Evident Integrity)"]
    end
```
