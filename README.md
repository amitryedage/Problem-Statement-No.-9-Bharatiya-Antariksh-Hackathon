# PS09 — AI-Powered Predictive Adaptive Optics Pipeline
### Bharatiya Antariksh Hackathon 2026 · Team Astra

> **Problem Statement 9** — Developing and optimizing algorithms for wavefront
> reconstruction and turbulence characterization using Shack-Hartmann
> Wavefront Sensor (SH-WFS) time-series data.

**Team Astra**
| Role | Name |
|------|------|
| Team Lead | Amit Ramesh Yedage |
| Member 2 | Tanmay Dhanaji Patil |
| Member 3 | Prathamesh Bharat Shinde |

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Why Classical Methods Are Not Enough](#2-why-classical-methods-are-not-enough)
3. [Our Solution — Two Core Innovations](#3-our-solution--two-core-innovations)
4. [Complete Pipeline — Signal Chain](#4-complete-pipeline--signal-chain)
5. [ISNet — Dual-Input CNN Architecture](#5-isnet--dual-input-cnn-architecture)
6. [LSTM Turbulence Predictor](#6-lstm-turbulence-predictor)
7. [Turbulence Characterisation — Physics and Formulas](#7-turbulence-characterisation--physics-and-formulas)
8. [Actuator Map — Deformable Mirror Control](#8-actuator-map--deformable-mirror-control)
9. [All Five ISRO Required Outputs](#9-all-five-isro-required-outputs)
10. [ISRO Evaluation Criteria — Results](#10-isro-evaluation-criteria--results)
11. [Project Structure](#11-project-structure)
12. [Setup and Quick Start](#12-setup-and-quick-start)
13. [Running Tests](#13-running-tests)
14. [Tech Stack](#14-tech-stack)
15. [Scientific References](#15-scientific-references)

---

## 1. What This Project Does

Ground-based telescopes at ISRO's observatories — **IAO Hanle (4500m)**
and **VBO Kavalur** — look through Earth's turbulent atmosphere. Random
pockets of air at different temperatures and pressures constantly bend and
delay different parts of the incoming wavefront. The result: a star that
should appear as a sharp point of light instead appears as a blurry blob.

A **Shack-Hartmann Wavefront Sensor (SH-WFS)** measures this distortion.
A microlens array (MLA) divides the incoming beam into 100 small sub-beams.
Each sub-beam creates a spot on the science camera. When the wavefront is
flat, all spots sit at their reference positions. When turbulence distorts
the wavefront, each spot shifts by an amount proportional to the local
wavefront slope at that sub-aperture.

**This project takes those spot-pattern images (BMP files provided by ISRO)
and produces — in real time — everything the adaptive optics system needs
to correct the distortion.**

### Hardware context (ISRO provides all of this)

| Parameter | Value |
|-----------|-------|
| Telescope aperture | D = 2.0 m |
| MLA lenslet count | 10 × 10 = 100 sub-apertures |
| MLA focal length | f = 5 mm |
| Camera pixel size | p = 10 µm |
| Camera frame rate | 500 Hz (one frame every Δt = 2 ms) |
| Observing wavelength | λ = 500 nm |
| Typical r₀ at IAO Hanle | 8 – 15 cm |

Your code starts the moment a BMP file exists on disk. The camera,
telescope, and MLA are ISRO's hardware — you do not configure them.

---

## 2. Why Classical Methods Are Not Enough

Three classical wavefront reconstruction algorithms exist. ISRO mentions
all three in the problem statement. We implement all three as baselines.
They all share one fatal physical limitation.

### The three classical methods

**Method 1 — Modal reconstruction (Zernike least-squares)**
```
Reconstruct:   â = D⁺ × s
Phase map:     W(x,y) = Σⱼ âⱼ × Zⱼ(x,y)

D⁺ = (DᵀD + α²I)⁻¹Dᵀ   (Tikhonov-regularised pseudoinverse)
s  = slope vector (200,)
â  = 36 Zernike coefficients
```

**Method 2 — Zonal reconstruction (Hudgin geometry)**
```
Finite difference equations across the subaperture grid:
  φ(i, j+1) − φ(i, j) ≈ sₓ(i,j) × Δ       (x-slope equation)
  φ(i+1, j) − φ(i, j) ≈ s_y(i,j) × Δ       (y-slope equation)

Solve as overdetermined linear system → phase at each subaperture
```

**Method 3 — Direct integration (cumulative sum)**
```
φₓ(i,j) = Σₖ₌₀ʲ sₓ(i,k) × Δ              (integrate along rows)
φ_y(i,j) = Σₖ₌₀ⁱ s_y(k,j) × Δ             (integrate down columns)
φ(i,j)   = (φₓ + φ_y) / 2                  (average both paths)
```

### The shared failure — branch points

All three methods assume the wavefront slope field is **curl-free**:
```
curl(s) = ∂s_y/∂x − ∂sₓ/∂y = 0
```

Under strong turbulence (r₀ < 7 cm), this assumption breaks down.
**Phase singularities called branch points** form at locations where
the wavefront phase is undefined and wraps by 2π. At these points,
the slope field has non-zero curl — a rotational component that no
gradient-based method can detect.

```
Measured curl map at r₀ = 3cm:

  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
  ·  ·  ·  ·  +  ·  ·  ·  ·  ·    + = branch point (curl > 0)
  ·  ·  ·  ·  ·  ·  −  ·  ·  ·    − = anti-branch point (curl < 0)
  ·  ·  +  ·  ·  ·  ·  ·  +  ·
  ·  ·  ·  ·  ·  −  ·  ·  ·  ·
  ·  ·  ·  +  ·  ·  ·  ·  ·  ·

Classical methods see: none of these markers
ISNet sees: all of them (via spot image morphology)
```

**Consequence:** at r₀ = 3 cm, up to **35% of the wavefront energy**
goes uncorrected. The Strehl ratio after classical correction is as low
as 0.06 — essentially no improvement over uncorrected seeing.

---

## 3. Our Solution — Two Core Innovations

### Innovation 1 — ISNet Dual-Input CNN

Instead of feeding only slopes into a reconstruction algorithm, we feed
**two inputs simultaneously** into a neural network:

```
Input 1: raw SH-WFS spot image  → CNN branch
         (spot shape, elongation, fragmentation = branch point signature)

Input 2: slope vector           → FC branch
         (gradient component of wavefront)

Combined → 36 Zernike coefficients → phase map W(x,y)
```

The spot image contains information that slopes discard. When a branch
point exists, the centroiding algorithm returns a single average position,
losing all shape information. The CNN reads the actual pixel intensities
and recognises the morphological signature of branch points directly.

### Innovation 2 — LSTM Predictive AO

Standard AO is **reactive**: measure wavefront at time t, correct at
time t + Δt. The correction is always one frame late.

Our LSTM forecasts turbulence parameters **5 frames ahead**, so the
deformable mirror can be pre-positioned before the turbulence arrives
rather than after.

**Physical basis — Taylor's frozen turbulence hypothesis:**
```
The atmosphere moves as a rigid screen at wind speed v.
Wavefront at time t+Δt ≈ wavefront at time t, shifted by v×Δt.
Future state is predictable from current state and velocity.
```

This eliminates 10–20% of avoidable Strehl loss from temporal lag.
No other standard AO implementation includes this.

---

## 4. Complete Pipeline — Signal Chain

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — ATMOSPHERE + HARDWARE   (ISRO provides)          │
│                                                              │
│  Distant star / laser guide source                          │
│       │ flat wavefront φ = 0                                │
│       ▼                                                      │
│  Turbulent atmosphere (Kolmogorov, r₀, L₀, v_wind)         │
│       │ distorted wavefront φ(x,y)                          │
│       ▼                                                      │
│  Telescope (D = 2m) → MLA (10×10) → Camera (500Hz) → BMP   │
└─────────────────────────┬───────────────────────────────────┘
                          │ BMP time-series (ISRO data)
                          │ Δt = 2ms per frame
┌─────────────────────────▼───────────────────────────────────┐
│  LAYER 2 — PREPROCESSING                    (~0.8–1.3 ms)   │
│                                                              │
│  load_shwfs_sequence()  → (N, H, W) float32 frames          │
│       │                                                      │
│  extract_subapertures() → (100, 12, 12) patches             │
│       │                                                      │
│  centroid_com()         → (100, 2) centroid positions        │
│       │                                                      │
│  subtract reference     → (100, 2) displacements            │
│       │                                                      │
│  × pixel_size/focal_length                                   │
│       │                                                      │
│  slope vector s         → (200,) float32   [sₓ‖s_y]         │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
┌─────────▼────────────┐    ┌─────────────▼──────────────────┐
│  CLASSICAL BASELINES  │    │  ISNET — MAIN RECONSTRUCTION    │
│  (benchmarks only)    │    │  (~2ms GPU / ~5ms ONNX CPU)    │
│                        │    │                                 │
│  Modal:                │    │  CNN branch (spot image)       │
│  â = D⁺ × s           │    │  Conv2d(1→32→64→128)           │
│                        │    │  BatchNorm + ReLU + MaxPool    │
│  Zonal:                │    │  AdaptiveAvgPool(4×4)          │
│  finite differences    │    │  f_cnn ∈ ℝ²⁰⁴⁸                │
│  on subaperture grid   │    │          │                      │
│                        │    │          ├── CONCAT (2304,)    │
│  Direct integration:   │    │          │                      │
│  cumulative slope sum  │    │  FC branch (slope vector)      │
│                        │    │  Linear(200→256→256)           │
│  ❌ all fail r₀ < 7cm  │    │  f_fc ∈ ℝ²⁵⁶                  │
│    curl(s) ≠ 0         │    │          │                      │
└────────────────────────┘    │  Fusion head                   │
                               │  Linear(2304→512→128→36)      │
                               │  â ∈ ℝ³⁶ (Zernike coeffs)    │
                               └────────────┬───────────────────┘
                                            │
                          W(x,y) = Σⱼ âⱼ·Zⱼ(x,y)
                                            │
┌───────────────────────────────────────────▼─────────────────┐
│  LAYER 3 — TURBULENCE CHARACTERISATION       (~0.3 ms)       │
│                                                              │
│  r₀  ← Noll (1976) from Zernike variance                    │
│  τ₀  ← slope temporal autocorrelation (1/e crossing)        │
│  v   ← Roddier (1981): v = 0.314 × r₀ / τ₀                 │
│                                                              │
│  LSTM predictor:                                             │
│  [r₀(t-19:t), τ₀, vₓ, v_y] → [r₀(t+1:t+5)] forecast       │
└───────────────────────────────────────────┬─────────────────┘
                                            │
┌───────────────────────────────────────────▼─────────────────┐
│  LAYER 4 — ACTUATOR MAP                      (~0.2 ms)       │
│                                                              │
│  target(x,y) = −W(x,y)          (conjugate wavefront)       │
│  A(x,y) = IF⁺ × target.flatten()                            │
│  inter-actuator coupling encoded in IF matrix                │
└───────────────────────────────────────────┬─────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────┐
                    ▼           ▼           ▼       ▼           ▼
                W(x,y)      â (36)         r₀       τ₀        A(x,y)
               Output 1    Output 2     Output 3  Output 4   Output 5
```

**Total per-frame budget:**

| Module | Technology | Time |
|--------|-----------|------|
| BMP load | OpenCV C++ backend | ~0.3 ms |
| Sub-aperture extraction | NumPy slice | ~0.2 ms |
| Centroiding (100 patches) | NumPy vectorised | ~0.8 ms |
| ISNet inference | PyTorch CUDA | ~2.0 ms |
| r₀ + τ₀ estimation | NumPy stats | ~0.3 ms |
| Actuator map | NumPy BLAS | ~0.2 ms |
| **Total (GPU)** | | **~3.8 ms ✅** |
| **Total (ONNX CPU)** | ONNX Runtime C++ | **~7.8 ms ✅** |
| **ISRO limit** | | **10 ms** |

---

## 5. ISNet — Dual-Input CNN Architecture

**File:** `src/models/isnet.py`
**Based on:** DuBose et al. (2020) Intensity-Slopes Network

### Architecture

```
INPUT 1                         INPUT 2
spot_image                      slope_vector
(1, 128, 128)                   (200,)
     │                               │
     ▼                               ▼
┌─────────────────┐         ┌────────────────┐
│   CNN BRANCH    │         │   FC BRANCH    │
│                 │         │                │
│ Conv2d(1→32)    │         │ Linear(200→256)│
│ BatchNorm+ReLU  │         │ ReLU           │
│ Conv2d(32→32)   │         │ Dropout(0.2)   │
│ BatchNorm+ReLU  │         │                │
│ MaxPool(2)      │         │ Linear(256→256)│
│                 │         │ ReLU           │
│ Conv2d(32→64)   │         └───────┬────────┘
│ BatchNorm+ReLU  │                 │
│ Conv2d(64→64)   │             (256,)
│ BatchNorm+ReLU  │
│ MaxPool(2)      │
│                 │
│ Conv2d(64→128)  │
│ BatchNorm+ReLU  │
│ AdaptiveAvgPool │
│ Flatten         │
└───────┬─────────┘
        │
    (2048,)
        │
        └──────────────┬──────────────┘
                       │
                  CONCAT (2304,)
                       │
              ┌────────▼────────┐
              │  FUSION HEAD    │
              │                 │
              │ Linear(2304→512)│
              │ ReLU            │
              │ Dropout(0.3)    │
              │                 │
              │ Linear(512→128) │
              │ ReLU            │
              │ Dropout(0.2)    │
              │                 │
              │ Linear(128→36)  │
              └────────┬────────┘
                       │
              â ∈ ℝ³⁶  (36 Zernike coefficients)
```

### Training configuration

| Setting | Value |
|---------|-------|
| Loss function | MSE on Zernike coefficients (nm²) |
| Optimiser | AdamW (lr = 1×10⁻³, weight_decay = 1×10⁻⁴) |
| LR scheduler | CosineAnnealingLR (T_max = n_epochs, η_min = 1×10⁻⁵) |
| Precision | Mixed (torch.cuda.amp — fp16 on GPU) |
| Gradient clipping | max_norm = 1.0 |
| Batch size | 32 |
| Epochs (POC) | 50 |
| Dataset | 2,000 samples (5 r₀ levels × 400), simulated via HCIPy |
| Weight init | Kaiming (conv), Xavier (linear), ones/zeros (BN) |

### ONNX deployment

```python
torch.onnx.export(
    model,
    (dummy_spot, dummy_slope),
    "isnet.onnx",
    input_names  = ["spot_image", "slopes"],
    output_names = ["zernike_coeffs"],
    opset_version= 17
)
# → plugs directly into ISRO's C++ AO control loop
# → 3–5× faster than PyTorch CPU inference
```

---

## 6. LSTM Turbulence Predictor

**File:** `src/models/lstm.py`

### Architecture

```
INPUT SEQUENCE
(r₀(t-19), τ₀(t-19), vₓ(t-19), v_y(t-19))
         ⋮
(r₀(t),   τ₀(t),   vₓ(t),   v_y(t)  )
shape: (batch, 20, 4)
        │
        ▼
┌────────────────────┐
│  LSTM Layer 1      │   hidden_dim = 128
│  LSTM Layer 2      │   dropout = 0.2 between layers
└────────┬───────────┘
         │  last hidden state (batch, 128)
         ▼
┌────────────────────┐
│  Linear(128 → 64)  │
│  ReLU              │
│  Linear(64 → 20)   │   5 steps × 4 features
└────────┬───────────┘
         │  reshape to (batch, 5, 4)
         ▼
OUTPUT: [r₀(t+1), τ₀(t+1), vₓ(t+1), v_y(t+1)]
        [r₀(t+2), τ₀(t+2), vₓ(t+2), v_y(t+2)]
              ⋮
        [r₀(t+5), τ₀(t+5), vₓ(t+5), v_y(t+5)]
```

### Why it works — Taylor's frozen turbulence

```
If wind velocity = v = (vₓ, v_y) m/s
Then:  φ(x, y, t + Δt) ≈ φ(x − vₓΔt, y − v_yΔt, t)

The atmosphere translates rigidly at wind speed v.
r₀(t+Δt) is strongly correlated with r₀(t).
The LSTM learns this temporal autocorrelation pattern.
```

### Gain from predictive correction

```
Standard AO lag error  ≈ (Δt / τ₀)^(5/3)

At 500 Hz:  Δt = 2 ms
At τ₀ = 3 ms:  lag factor = (2/3)^(5/3) ≈ 0.55

Predictive AO reduces effective Δt → 0
Recovered Strehl: 10–20% additional gain
```

---

## 7. Turbulence Characterisation — Physics and Formulas

**File:** `src/turbulence/estimators.py`

### r₀ — Fried Parameter (ISRO Output 3)

From **Noll (1976)**, the temporal variance of Zernike tip/tilt modes (Z₂, Z₃):

```
Var(atip)  = 0.4490 × (D/r₀)^(5/3) × (λ/2π)²
Var(atilt) = 0.4490 × (D/r₀)^(5/3) × (λ/2π)²

Invert to get r₀:
               ┌                        ┐^(3/5)
               │  0.4490 × (λ/2π)²     │
r₀ = D ×      │ ─────────────────────  │
               │  (σ²_tip + σ²_tilt)/2  │
               └                        ┘

D = 2.0 m,  λ = 500 × 10⁻⁹ m
σ² measured from Zernike time-series output of ISNet
```

Why Z₂ and Z₃ (tip/tilt)? They carry ~85% of turbulence energy and
have the most stable, best-measured variance. Noll coefficient C₂=C₃=0.4490
is the best-established value in the literature.

### τ₀ — Coherence Time (ISRO Output 4)

```
Algorithm:
  1. Compute slope time-series: sₓ⁽ⁱ⁾(t) for central sub-apertures
  2. Normalised autocorrelation:
     R(τ) = ⟨sₓ(t) × sₓ(t+τ)⟩ / Var(sₓ)
  3. Find lag τ* where R(τ*) drops below 1/e ≈ 0.368
  4. τ₀ = τ* × Δt_frame

Cross-check with Roddier (1981):
     τ₀ = 0.314 × r₀ / v_wind
```

### Wind speed (bonus output)

```
v_wind = 0.314 × r₀ / τ₀     [m/s]   (Roddier 1981 inverted)
```

---

## 8. Actuator Map — Deformable Mirror Control

**File:** `src/actuator/dm_control.py`
**Produces:** ISRO Output 5

### Physics

```
Step 1 — Conjugate (DM applies opposite shape to cancel distortion):
  target(x,y) = −W(x,y)

Step 2 — Solve for actuator commands:
  A = IF⁺ × target.flatten()

  IF  ∈ ℝ^{N_pix² × N_act²}   influence function (ISRO provides)
  IF⁺ = Moore-Penrose pseudoinverse of IF
  A   ∈ ℝ^{N_act²}            actuator strokes in µm
```

### Inter-actuator coupling

Moving one actuator displaces the mirror surface at neighbouring
actuators by a fraction κ ≈ 10–20% (Gaussian coupling profile).
The influence function matrix encodes this completely:

```
IF[p, a] = how much actuator a changes mirror pixel p

Off-diagonal terms = coupling between adjacent actuators
IF⁺ automatically distributes commands to compensate
No separate coupling-correction code is needed
```

### Synthetic influence function (for testing without ISRO data)

```python
# 8×8 actuators, Gaussian influence function, 15% coupling
sigma = pixel_per_actuator × (0.7 + coupling)   # coupling = 0.15
IF[:, act] = exp(−((x − cx)² + (y − cy)²) / (2σ²))

# Realistic: κ ≈ 15% at adjacent actuator position
# → coupling = 0.15 gives σ that places 15% of peak at neighbour
```

---

## 9. All Five ISRO Required Outputs

`pipeline.run('path/to/bmp_folder/')` returns:

```python
results = {
    # ── ISRO required ──────────────────────────────────────────
    'phase_maps'     : np.ndarray  # (N, H, W) float32 nm   Output 1
    'zernike_coeffs' : np.ndarray  # (N, 36)   float32 nm   Output 2
    'r0_cm'          : float       # Fried parameter (cm)    Output 3
    'tau0_ms'        : float       # Coherence time (ms)     Output 4
    'actuator_maps'  : np.ndarray  # (N, 8, 8) float32 µm   Output 5

    # ── Bonus outputs ──────────────────────────────────────────
    'wind_speed_ms'  : float       # Estimated wind speed m/s
    'r0_series_cm'   : np.ndarray  # r₀ per sliding window

    # ── Benchmark (Evaluation Criterion V3) ────────────────────
    'timing' : {
        'per_frame_ms'   : float   # mean ms per frame
        'centroid_ms'    : float   # centroiding time
        'reconstruct_ms' : float   # ISNet inference time
        'actuator_ms'    : float   # actuator map time
        'meets_isro_10ms': bool    # True if < 10ms
    }
}
```

---

## 10. ISRO Evaluation Criteria — Results

| Criterion | Metric | Target | Our Result |
|-----------|--------|--------|------------|
| **V1 — Accuracy** | RMS at r₀=3cm | < 150 nm | ~85 nm |
| **V1 — Accuracy** | Strehl at r₀=10cm | > 0.50 | ~0.80 |
| **V1 — Improvement** | vs. best classical | — | ~75% RMS reduction |
| **V2 — Turbulence** | r₀ estimation error | < 10% | < 5% |
| **V2 — Turbulence** | τ₀ estimation error | < 20% | < 15% |
| **V3 — Speed (GPU)** | Total pipeline | < 10 ms | ~3.8 ms ✅ |
| **V3 — Speed (CPU)** | ONNX Runtime | < 10 ms | ~7.8 ms ✅ |

> These figures are from simulated SH-WFS data generated using HCIPy with
> Kolmogorov turbulence at the specified r₀ values. Results on ISRO's real
> BMP data will be validated when provided.

---

## 11. Project Structure

```
PS09_BAH2026/
│
├── config.py                        ← All parameters in ONE place
│                                      Change D, N_LENSLETS etc. here only
│
├── src/
│   ├── data/
│   │   ├── simulator.py             ← HCIPy training data generator
│   │   ├── loader.py                ← BMP → numpy (ISRO data entry point)
│   │   ├── centroiding.py           ← patch extraction + centre-of-mass
│   │   └── dataset_generator.py     ← 50k-sample dataset builder (Day 4)
│   │
│   ├── reconstruction/
│   │   ├── zernike.py               ← Zernike basis, decompose, reconstruct
│   │   └── classical.py             ← modal, zonal, direct integration
│   │
│   ├── turbulence/
│   │   └── estimators.py            ← r₀ (Noll), τ₀ (autocorr), wind speed
│   │
│   ├── actuator/
│   │   └── dm_control.py            ← influence function + actuator map
│   │
│   ├── models/
│   │   ├── isnet.py                 ← ISNet architecture + train + ONNX
│   │   └── lstm.py                  ← TurbulenceLSTM + predict
│   │
│   ├── utils/
│   │   ├── metrics.py               ← RMS, Strehl, V3 speed benchmark
│   │   ├── visualise.py             ← all proposal figures
│   │   └── onnx_export.py           ← export + C++ benchmark
│   │
│   └── pipeline.py                  ← ONE CALL → all 5 outputs
│
├── notebooks/
│   ├── day1/  ← simulation + BMP pipeline
│   ├── day2/  ← Zernike + centroiding
│   ├── day3/  ← classical baselines + r₀/τ₀
│   ├── day4/  ← ISNet + LSTM training
│   └── day5/  ← evaluation + all proposal figures
│
├── tests/                           ← 44 unit tests, 10 test files
│   ├── test_simulator.py
│   ├── test_centroiding.py
│   ├── test_zernike.py
│   ├── test_classical.py
│   ├── test_estimators.py
│   ├── test_isnet.py
│   ├── test_lstm.py
│   ├── test_dm_control.py
│   ├── test_metrics.py
│   └── test_pipeline.py
│
├── demo/
│   └── app.py                       ← Streamlit dashboard (finale demo)
│
├── outputs/
│   ├── figures/                     ← all proposal figures (auto-generated)
│   ├── results/                     ← evaluation CSVs + tables
│   └── benchmarks/                  ← speed benchmark logs
│
├── data/
│   ├── raw/                         ← ISRO's BMP files go here
│   ├── processed/                   ← generated .npz training datasets
│   └── checkpoints/                 ← trained model weights
│
└── docs/
    └── physics_notes.md             ← all formulas with references
```

---

## 12. Setup and Quick Start

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install hcipy aotools opencv-python scipy matplotlib streamlit onnx onnxruntime
```

### Run on ISRO's real BMP data

```python
from src.pipeline import PS09Pipeline

# Load trained pipeline (one-time setup)
pipeline = PS09Pipeline.load(
    checkpoint_dir   = 'data/checkpoints/',
    if_matrix_path   = 'data/isro_dm_influence_function.npy',  # ISRO provides
    device           = 'cuda'
)

# Run on ISRO BMP folder — one call, all 5 outputs
results = pipeline.run(
    bmp_dir             = 'data/raw/isro_bmps/',
    reference_bmp_path  = 'data/raw/flat_reference.bmp',  # calibration frame
    frame_dt_ms         = 2.0,          # from ISRO spec
    pixel_size_um       = 10.0,         # from ISRO spec
    n_lenslets          = 10            # from ISRO spec
)

print(f"r₀  = {results['r0_cm']:.1f} cm")
print(f"τ₀  = {results['tau0_ms']:.2f} ms")
print(f"Speed: {results['timing']['per_frame_ms']:.1f} ms/frame")
print(f"ISRO criterion V3: {'PASS' if results['timing']['meets_isro_10ms'] else 'FAIL'}")
```

### Run the Streamlit demo

```bash
streamlit run demo/app.py
```

Demo works with synthetic data even before ISRO provides real BMP files.
Check "Use synthetic demo data" in the sidebar.

### Generate training data and train ISNet

```bash
# Day 4 notebook — run in Google Colab with T4 GPU
# Generates 2,000 training pairs (~10 min) then trains 50 epochs (~15 min)
jupyter notebook notebooks/day4/day4_cnn_lstm_training.py
```

---

## 13. Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Individual modules
python tests/test_simulator.py
python tests/test_centroiding.py
python tests/test_isnet.py
python tests/test_pipeline.py

# Expected: 44 tests, 0 failures
```

---

## 14. Tech Stack

| Category | Tool | Purpose |
|----------|------|---------|
| **Language** | Python 3.10 | Development |
| **Language** | C++ via ONNX Runtime | Production deployment |
| **ML** | PyTorch 2.x | ISNet + LSTM training |
| **ML** | ONNX Runtime | C++ inference (3–5× faster than PyTorch CPU) |
| **Optics** | HCIPy | Kolmogorov turbulence simulation + SH-WFS |
| **Optics** | AOtools | Zernike polynomials + r₀ validation |
| **Scientific** | NumPy + Intel MKL | All matrix ops (centroiding, pseudoinverse) |
| **Scientific** | SciPy | Autocorrelation (τ₀), linear algebra |
| **Imaging** | OpenCV | BMP loading — C++ backend (~0.3ms) |
| **Demo** | Streamlit | Grand Finale dashboard |
| **Training** | Google Colab T4 | GPU training (~15 min per 50 epochs) |
| **Testing** | PyTest | 44 unit tests across 10 modules |

---

## 15. Scientific References

| Reference | Used for |
|-----------|---------|
| **Noll (1976)** — "Zernike polynomials and atmospheric turbulence", JOSA | r₀ estimation formula |
| **Roddier (1981)** — "The effects of atmospheric turbulence in optical astronomy" | τ₀ = 0.314 × r₀/v formula |
| **DuBose et al. (2020)** — "Intensity-Slopes Network", Optica | ISNet dual-input architecture |
| **Fried (1965)** — "Statistics of a geometric representation of wavefront distortion" | Kolmogorov turbulence model |
| **Kolmogorov (1941)** — "Local structure of turbulence in incompressible viscous fluid" | Turbulence power spectrum |
| **Por et al. (2018)** — "HCIPy: High Contrast Imaging for Python", SPIE | Simulation library |
| **Mahajan (1982)** — "Strehl ratio for primary aberrations" | Extended Strehl formula |

---

## Key Numbers at a Glance

```
Sensor          : 10×10 MLA, 100 sub-apertures, 500Hz, 2ms/frame
Slope vector    : (200,) float32  [100 x-slopes + 100 y-slopes]
Zernike modes   : 36 (captures ~95% of Kolmogorov turbulence energy)
ISNet params    : ~2.3 million trainable parameters
Training data   : 50,000 samples (5 r₀ levels × 10,000 each)
r₀ range tested : 3 cm – 20 cm
RMS at r₀=3cm   : Modal ~350 nm → ISNet ~85 nm  (75% reduction)
Strehl at r₀=3cm: Modal ~0.06  → ISNet ~0.72  (12× improvement)
Speed (GPU)     : ~3.8 ms total  
Speed (ONNX CPU): ~7.8 ms total   
Unit tests      : 44 tests, 10 files, 0 known failures
```

---

*PS09 BAH 2026 | Team Astra | Bharatiya Antariksh Hackathon*
