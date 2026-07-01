# PS09 — AI-Powered Predictive Adaptive Optics
### Bharatiya Antariksh Hackathon 2026 | Team Astra

**Problem Statement 9:** Developing and optimizing algorithms for wavefront reconstruction and turbulence characterization using Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data.

**Team Astra** — Amit Ramesh Yedage (Lead), Tanmay Dhanaji Patil, Prathamesh Bharat Shinde 

---

## 1. The Problem.

A ground telescope looking through the atmosphere sees a blurred image because turbulent air constantly distorts incoming starlight. A Shack-Hartmann Wavefront Sensor (SH-WFS) measures that distortion using a microlens array (MLA) that splits the incoming beam into ~100 spots on a camera; how those spots shift tells you the shape of the distorted wavefront.

The job of this project is to take a stream of SH-WFS camera frames and, in real time, produce everything a telescope's adaptive optics (AO) system needs to correct for the distortion — fast enough to matter (ISRO's target: **under 10 ms per frame**).

---

## 2. Core Idea.

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

**Speed budget :** data ingestion ~0.5 ms + centroiding ~0.8 ms + reconstruction (GPU ~2 ms / CPU ~5 ms) → total **< 8–10 ms**, meeting the ISRO V3 speed criterion.

---

## 4. The Five ISRO-Required Outputs

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

## 5. ISNet — the Reconstruction Model 

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

## 6. TurbulenceLSTM — the Predictive Layer

`src/models/lstm.py` — 2-layer LSTM (hidden dim 128) with a small FC output head.

- **Input:** last 20 frames of `(r0, τ0, wind_x, wind_y)`
- **Output:** forecast of the next 5 frames of the same 4 values
- **Physics basis:** Taylor's frozen-turbulence hypothesis — the atmosphere is treated as a fixed pattern being blown past the telescope by wind, so its future state is *predictable* from its current state and velocity
- **Purpose:** eliminates the "temporal lag" problem in classical reactive AO, where the mirror always corrects for what turbulence *was* one frame ago rather than what it *is now*

---

## 7. Turbulence Physics — the Formulas Actually Implemented

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

## 8. Classical Baselines (why they exist, and why they're not enough)

`src/reconstruction/classical.py` implements three traditional reconstruction methods purely as **benchmarks**, not as the main solution:

| Method | How | Weakness |
|---|---|---|
| **Modal** | `D_pinv @ slopes` → Zernike coefficients (one matmul) | Assumes zero curl; per the code's own comment, can miss **up to 35% of wavefront energy** at r0 = 3 cm |
| **Zonal** | Hudgin finite-difference geometry | Same curl blindness, different numerical approach |
| **Direct integration** | Cumulative sum of slopes | Fastest, least accurate |

This is honest framing on the team's part: the classical methods are shown to **fail exactly where it matters** (strong turbulence), which is the entire justification for building ISNet.

---

## 9. Actuator / Deformable Mirror Control

`src/actuator/dm_control.py`:
- The DM correction is the **conjugate** of the measured phase: `target = -phase_nm`.
- Actuator commands are solved as `A = IF_pinv @ target`, where `IF` is the influence function.
- **Inter-actuator coupling** (moving one actuator slightly moves its neighbours) is captured entirely inside the influence-function matrix — no separate coupling logic is needed.
- Since ISRO's real influence-function data isn't available yet, the repo includes a **synthetic Gaussian influence function generator** (8×8 actuators, ~15% coupling, a typical real-DM value) so the pipeline can be developed and tested end-to-end before real hardware data arrives.

---

## 10. Evaluation Criteria vs. Claimed Status

From the architecture slide's summary table:

| Criterion | Metric | Claimed Status |
|---|---|---|
| **V1 — Accuracy** | RMS < 50 nm / Strehl > 0.70 | Met (claimed 75% better than baseline model) |
| **V2 — Turbulence** | r0 error < 5% / τ0 error < 15% | Met (Noll 1976 + autocorrelation methods) |
| **V3 — Speed** | GPU ~5 ms / CPU ~8 ms | Met (< 10 ms limit) |

⚠️ **Read this table as the pitch's claims, not independently verified results** — see Section 12 for what I could actually confirm by running the code.

---

## 11. Tech Stack (as declared in the pitch)

| Category | Tools |
|---|---|
| Languages & Core | Python 3.10, C++ (via ONNX Runtime), NumPy + Intel MKL |
| ML / Modelling | PyTorch (ISNet CNN, LSTM), ONNX Runtime, mixed-precision training (AMP) |
| Optics & Simulation | HCIPy (turbulence + SH-WFS simulation), AOtools (Zernike / r0 utilities), SciPy (autocorrelation, linalg) |
| Data & Imaging | OpenCV (BMP I/O, centroiding), Astropy (FITS support), Matplotlib (diagnostics) |
| Deployment & Demo | Streamlit (dashboard — *see gap below*), ONNX (C++ AO-loop integration), Google Colab (T4 GPU training) |
| Testing & Tooling | PyTest (44 unit tests claimed), Git, GitHub Actions (*planned*, not yet present) |

---

## 12. What I Actually Verified in the Repository (ground truth, not the pitch)

I cloned the repo and ran it directly rather than just reading the slides. Here's what's real vs. aspirational:

**✅ Solid and real:**
- 59 real git commits showing genuine iterative development (centroiding → Zernike → classical reconstruction → estimators → ISNet training → LSTM → ONNX export → bug fixes), matching the "Day 1–7" narrative in the pitch.
- The physics formulas in `estimators.py` match standard literature (Noll 1976, Roddier 1981) — not hand-waved.
- `pytest` run in this environment: **18 passed / 8 failed** out of the test suite. The 8 failures were environment issues (aotools incompatible with NumPy 2.0's removed `np.math`, and `hcipy` not installed) — not necessarily bugs in the team's logic, but it does mean the test suite isn't currently green out-of-the-box on a fresh environment.
- Inline bug-fix comments in the code itself (e.g. `estimators.py`: *"if no valid subapertures found, return sentinel -1.0 instead of nan so callers can detect the failure cleanly"*) show real debugging, not just a first draft.

**❌ Gaps between the pitch and the repo — stated plainly:**
- **`README.md` is completely empty (0 bytes).** No setup instructions exist despite the pitch's "Documentation" bullet on the GitHub QR slide.
- **`demo/app.py` is empty (0 functional lines).** The pitch's "Streamlit dashboard" deployment claim has no corresponding code yet.
- **`src/data/dataset_generator.py` is empty (0 lines)** — despite being imported conceptually as part of the data pipeline.
- **No `requirements.txt` or environment/dependency file anywhere** in the repo, despite depending on PyTorch, aotools, hcipy, OpenCV, ONNX, SciPy — a fresh clone can't be set up without guessing versions.
- **Notebooks `day2` through `day7` are empty placeholder `README.md` files, not actual notebooks.** Only `day1` has a real `.ipynb`. So the "week-long build log" visual in the pitch overstates what's actually browsable in `notebooks/`.
- **Stray `node_modules/` and `package.json`** exist in an otherwise pure-Python project, and the `package.json` lists `"numpy": "^0.0.1"` as an npm dependency — which is meaningless (numpy isn't an npm package). This looks like leftover/junk scaffolding, not intentional infrastructure.
- **`config.py` hardcodes `DEVICE = 'cuda'`** — will throw on any machine without a GPU unless overridden elsewhere in code not shown here.
- **`__pycache__` is committed to git** in two places — should be gitignored.
- I could not verify the **44 unit tests / GitHub Actions CI** claims from the tech-stack slide directly — the repo currently has **6 test files, ~26 test functions total** (`test_centroiding.py`, `test_classical.py`, `test_estimators.py`, `test_metrics.py`, `test_simulator.py`, `test_zernike.py`), and there is no CI config file (`.github/workflows/`) in the repo at all — so "GitHub Actions (planned CI)" is accurate as *planned*, not yet implemented.

**Bottom line:** the physics and model design (ISNet dual-input, LSTM prediction, classical baselines as honest comparison points) are genuinely well thought out and the core reconstruction/estimation code works. What's missing is everything around it — docs, a working demo, dependency pinning, and a couple of empty placeholder files — which is normal for a hackathon-stage project but worth fixing before anyone else tries to run it.


3. Either implement `demo/app.py` minimally or drop the Streamlit claim from the pitch until it exists.
4. Remove `node_modules/`, the bogus `package.json`, and `__pycache__` from version control (add a `.gitignore`).
5. Make `DEVICE` in `config.py` auto-detect (`'cuda' if torch.cuda.is_available() else 'cpu'`) so it doesn't break on non-GPU judge machines.
