# PS09 — AI-Powered Predictive Adaptive Optics

### Bharatiya Antariksh Hackathon 2026 · Team Astra

**Problem Statement 9:** Developing and optimizing algorithms for wavefront reconstruction and turbulence characterization using Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data.

**Team Astra** — Amit Ramesh Yedage (Lead) · Tanmay Dhanaji Patil · Prathamesh Bharat Shinde

---

## 1. The Problem

A ground telescope looking through the atmosphere sees a blurred image because turbulent air constantly distorts incoming starlight. A Shack-Hartmann Wavefront Sensor (SH-WFS) measures that distortion using a microlens array (MLA) that splits the incoming beam into ~100 spots on a camera; how those spots shift tells you the shape of the distorted wavefront.

The job of this project is to take a stream of SH-WFS camera frames and, in real time, produce everything a telescope's adaptive optics (AO) system needs to correct for the distortion — fast enough to matter (target: **under 10 ms per frame**).

---

## 2. Core Idea

> "We reconstruct the wavefront using **ISNet** — a dual-input CNN that reads both the SH-WFS spot image and the slope vector together, enabling it to recover the rotational component of distortion that single-input methods miss. A companion **LSTM** forecasts atmospheric turbulence a few frames ahead, enabling predictive rather than purely reactive correction."

Two ideas carry the whole project:

1. **Dual-input reconstruction** — classical AO only looks at spot *displacements* (slopes). This project also feeds the raw spot *image* into a CNN, because spot shape/fragmentation carries information (branch points / curl) that displacement alone throws away.
2. **Predictive correction** — instead of only reacting to the current frame, an LSTM forecasts where turbulence is heading a few frames ahead (using Taylor's frozen-turbulence hypothesis: the atmosphere behaves like a rigid pattern blown by wind), so the deformable mirror can be pre-positioned rather than always lagging one frame behind.

---

## 3. End-to-End Pipeline

```
                     ATMOSPHERE (Kolmogorov turbulence, r0, L0, wind)
                                     │
                          Starlight φ(x,y) distorted
                                     │
                     Microlens Array (10×10, f = 5mm) → 100 spots
                                     │
                    Science Camera — 500 Hz, 10 µm pixels, 2 ms/frame (BMP)
                                     │
        ┌───────────────────────────┼───────────────────────────────┐
        │                 PREPROCESSING (~0.5–1.3 ms)                │
        │  • BMP load + sub-aperture extraction (100 patches, 12×12) │
        │  • Centroiding → (100,2) → flattened (200,) slope vector   │
        └───────────────────────────┬───────────────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │            RECONSTRUCTION STAGE               │
              │                                               │
              │  Classical baselines (fast, fragile):         │
              │    • Modal (Zernike least-squares)             │
              │    • Zonal (Hudgin finite-difference)          │
              │    • Direct integration (cumulative sum)       │
              │    → fail when turbulence is strong (r0 < 7cm) │
              │      because branch points create "curl" that  │
              │      gradient-only methods can't see            │
              │                                               │
              │  ISNet (the actual innovation):                │
              │    CNN branch (spot image) ──┐                 │
              │                                ├─ fuse → 36     │
              │    FC branch (slope vector) ──┘   Zernike coeffs│
              └──────────────────────┬──────────────────────┘
                                     │
                 Zernike coefficients → phase map W(x,y)
                                     │
              ┌──────────────────────┴──────────────────────┐
              │        TURBULENCE CHARACTERIZATION            │
              │  • r0 (Fried parameter)  — Noll 1976 formula   │
              │    from tip/tilt (Z2, Z3) temporal variance     │
              │  • τ0 (coherence time)   — slope autocorrelation│
              │    (1/e crossing of temporal autocorrelation)   │
              │  • wind speed            — Roddier: v=0.314·r0/τ0│
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │         PREDICTIVE LAYER (LSTM)               │
              │  Input : last 20 frames of (r0, τ0, vx, vy)    │
              │  Output: next 5 frames forecast                │
              │  Physics basis: Taylor frozen-turbulence        │
              │  → lets the mirror pre-position instead of      │
              │    reacting a frame late                         │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │           ACTUATOR MAP GENERATION              │
              │  target = -phase_nm  (conjugate wavefront)      │
              │  A(x,y) = IF_pinv @ target                      │
              │  Influence function (Gaussian, ~15% coupling)   │
              │  encodes how neighbouring actuators pull on      │
              │  each other — no extra logic needed, it's        │
              │  baked into the pseudoinverse                    │
              └──────────────────────┬──────────────────────┘
                                     │
                        DEFORMABLE MIRROR COMMANDS
```

**Speed budget:** data ingestion ~0.5 ms + centroiding ~0.8 ms + reconstruction (GPU ~2 ms / CPU ~5 ms) → total **< 8–10 ms**.

---

## 4. The Five Required Outputs

`src/pipeline.py` is explicit about producing all five, plus two bonus outputs:

| # | Output | What it is |
|---|--------|-----------|
| 1 | `phase_maps` | Reconstructed 2D wavefront **W(x,y)** per frame |
| 2 | `zernike_coeffs` | 36 Zernike coefficients per frame |
| 3 | `r0_cm` | Fried parameter (turbulence strength) |
| 4 | `tau0_ms` | Coherence time (turbulence speed) |
| 5 | `actuator_maps` | Deformable-mirror stroke commands **A(x,y)** |
| bonus | `wind_speed_ms` | Estimated wind speed |
| bonus | `r0_series` | r0 over a sliding time window |
| + | `timing` | Per-module timing breakdown for the 10 ms benchmark |

---

## 5. ISNet — The Reconstruction Model

Based on the DuBose et al. (2020) Intensity-Slopes Network concept. Implemented in `src/models/isnet.py`.

**Why dual input, concretely:**
- Slopes alone measure the *gradient* of the wavefront → miss the *curl* component that branch points introduce.
- Spot images show *shape distortion* (elongation, fragmentation, asymmetry) that reveals branch points.
- Feeding both → reconstructs gradient **and** curl → claimed 30–60% better Strehl ratio in strong turbulence (per code comments).

**Architecture:**

```
Spot image (1,128,128) ──► CNN branch ──► 2048-d features ─┐
                            (Conv2d 32→64→128,               ├─► Fusion (2304→512→128→36)
                             BatchNorm, MaxPool,              │      = 36 Zernike coefficients
                             AdaptiveAvgPool)                 │
Slope vector (200,) ─────► FC branch ────► 256-d features ──┘
                            (Linear 256→256, Dropout 0.2)
```

- Weight init: Kaiming (conv layers), Xavier (linear layers)
- Training: AdamW + cosine annealing LR, mixed-precision (AMP), gradient clipping, best-checkpoint saving
- Deployment: exported to **ONNX** (opset 17) for a C++ inference path — claimed 3–5× faster than PyTorch on CPU

---

## 6. TurbulenceLSTM — The Predictive Layer

`src/models/lstm.py` — 2-layer LSTM (hidden dim 128) with a small FC output head.

- **Input:** last 20 frames of `(r0, τ0, wind_x, wind_y)`
- **Output:** forecast of the next 5 frames of the same 4 values
- **Physics basis:** Taylor's frozen-turbulence hypothesis — the atmosphere is treated as a fixed pattern being blown past the telescope by wind, so its future state is *predictable* from its current state and velocity
- **Purpose:** eliminates the "temporal lag" problem in classical reactive AO, where the mirror always corrects for what turbulence *was* one frame ago rather than what it *is now*

---

## 7. Turbulence Physics — The Formulas Actually Implemented

From `docs/physics_notes.md` and `src/turbulence/estimators.py`:

**Fried parameter r₀** (Noll 1976), estimated from the temporal variance of the tip/tilt Zernike modes (Z2, Z3):

```
Var(a_tip) = 0.4490 × (D/r0)^(5/3) × (λ/2π)²
```

**Coherence time τ₀** — found from the temporal autocorrelation of the slope signal; τ₀ is the lag at which the mean autocorrelation drops below 1/e.

**Wind speed** (Roddier 1981):

```
v_wind = 0.314 × r0 / τ0
```

The site constraint baked into the config reflects a real observatory: **IAO Hanle/VBO, D = 2 m aperture**, operating range r0 = 3–20 cm, τ0 = 1–10 ms.

---

## 8. Classical Baselines — Why They Exist, and Why They're Not Enough

`src/reconstruction/classical.py` implements three traditional reconstruction methods purely as **benchmarks**, not as the main solution:

| Method | How | Weakness |
|---|---|---|
| **Modal** | `D_pinv @ slopes` → Zernike coefficients (one matmul) | Assumes zero curl; per the code's own comment, can miss **up to 35% of wavefront energy** at r0 = 3 cm |
| **Zonal** | Hudgin finite-difference geometry | Same curl blindness, different numerical approach |
| **Direct integration** | Cumulative sum of slopes | Fastest, least accurate |

This is honest framing: the classical methods are shown to **fail exactly where it matters** (strong turbulence), which is the entire justification for building ISNet.

---

## 9. Actuator / Deformable Mirror Control

`src/actuator/dm_control.py`:

- The DM correction is the **conjugate** of the measured phase: `target = -phase_nm`.
- Actuator commands are solved as `A = IF_pinv @ target`, where `IF` is the influence function.
- **Inter-actuator coupling** (moving one actuator slightly moves its neighbours) is captured entirely inside the influence-function matrix — no separate coupling logic is needed.
- Since real influence-function data isn't available yet, the repo includes a **synthetic Gaussian influence function generator** (8×8 actuators, ~15% coupling, a typical real-DM value) so the pipeline can be developed and tested end-to-end before real hardware data arrives.

---

## 10. Evaluation Criteria vs. Claimed Status

| Criterion | Metric | Claimed Status |
|---|---|---|
| **V1 — Accuracy** | RMS < 50 nm / Strehl > 0.70 | Met (claimed 75% better than baseline model) |
| **V2 — Turbulence** | r0 error < 5% / τ0 error < 15% | Met (Noll 1976 + autocorrelation methods) |
| **V3 — Speed** | GPU ~5 ms / CPU ~8 ms | Met (< 10 ms limit) |

⚠️ These are the project's claimed results as presented, not independently benchmarked figures.

---

## 11. Tech Stack

| Category | Tools |
|---|---|
| Languages & Core | Python 3.10, C++ (via ONNX Runtime), NumPy + Intel MKL |
| ML / Modelling | PyTorch (ISNet CNN, LSTM), ONNX Runtime, mixed-precision training (AMP) |
| Optics & Simulation | HCIPy (turbulence + SH-WFS simulation), AOtools (Zernike / r0 utilities), SciPy (autocorrelation, linalg) |
| Data & Imaging | OpenCV (BMP I/O, centroiding), Astropy (FITS support), Matplotlib (diagnostics) |
| Deployment & Demo | Streamlit (dashboard), ONNX (C++ AO-loop integration), Google Colab (T4 GPU training) |
| Testing & Tooling | PyTest, Git, GitHub Actions (planned) |

---

## 12. Bottom Line

The physics and model design — ISNet's dual-input reconstruction, LSTM-based turbulence prediction, and classical baselines used as honest comparison points — are genuinely well thought out, and the core reconstruction/estimation code works. The architecture is coherent end-to-end: sensor frame in, deformable-mirror command out, with every required output produced along the way.

---
