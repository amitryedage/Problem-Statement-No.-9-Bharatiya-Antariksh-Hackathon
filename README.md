# 🔭 PS09 BAH 2026 — Wavefront Reconstruction & Turbulence Characterisation
**Team:** [Your Team Name] | **Hackathon:** Bharatiya Antariksh Hackathon 2026 | ISRO × Hack2Skill

---

## 🗂️ Project Structure

```
PS09_BAH2026/
│
├── config.py                    ← ALL parameters here. Change once, adapts everywhere.
│
├── notebooks/                   ← Daily POC work (Colab notebooks)
│   ├── day1/ ✅                 ← Wavefront physics + SH-WFS simulation
│   ├── day2/ 🔲                 ← Zernike + centroiding
│   ├── day3/ 🔲                 ← Classical baselines + r₀/τ₀
│   ├── day4/ 🔲                 ← ISNet CNN + LSTM training
│   ├── day5/ 🔲                 ← Evaluation + proposal figures
│   ├── day6/ 🔲                 ← Proposal writing
│   └── day7/ 🔲                 ← Submit
│
├── src/                         ← MAIN CODEBASE (production-ready)
│   ├── data/
│   │   ├── simulator.py  ✅     ← Wavefront + SH-WFS simulation engine
│   │   ├── loader.py     ✅     ← BMP file loader (ISRO real data)
│   │   ├── centroiding.py🔲    ← Spot → slope vector (Day 2)
│   │   └── dataset_generator.py← 50k sample generator (Day 4)
│   │
│   ├── reconstruction/
│   │   ├── zernike.py    🔲    ← Zernike basis + decomposition (Day 2)
│   │   └── classical.py  🔲    ← Modal + zonal + direct integration (Day 3)
│   │
│   ├── turbulence/
│   │   └── estimators.py 🔲    ← r₀ + τ₀ + wind speed (Day 3)
│   │
│   ├── actuator/
│   │   └── dm_control.py 🔲    ← Actuator map with coupling (Day 4)
│   │
│   ├── models/
│   │   ├── isnet.py      🔲    ← ISNet dual-input CNN (Day 4)
│   │   └── lstm.py       🔲    ← Turbulence LSTM predictor (Day 4)
│   │
│   ├── utils/
│   │   ├── metrics.py    ✅    ← RMS, Strehl, speed benchmark
│   │   └── visualise.py  ✅    ← All proposal figures
│   │
│   └── pipeline.py       🔲    ← Complete integrated pipeline (Day 5)
│
├── tests/
│   ├── test_simulator.py ✅
│   └── test_metrics.py   ✅
│
├── outputs/
│   ├── figures/                ← All proposal figures saved here
│   ├── results/                ← Evaluation CSVs, benchmark tables
│   └── benchmarks/             ← Speed benchmark logs
│
├── proposal/
│   ├── figures/                ← High-res figures for submission
│   └── draft/                  ← Section drafts (Day 6)
│
├── demo/
│   └── app.py          🔲     ← Streamlit finale demo (Day 5)
│
├── docs/
│   └── physics_notes.md ✅    ← Key formulas + references
│
└── data/
    ├── raw/                    ← ISRO's BMP files (when provided)
    ├── processed/              ← Generated .npz training datasets
    └── checkpoints/            ← Trained model weights
```

---

## 🎯 ISRO's 5 Required Outputs

| # | Output | Status | Module |
|---|--------|--------|--------|
| 1 | W(xi,yi) — Phase map per frame | 🔲 Day 4 | src/reconstruction/ |
| 2 | Zernike coefficients per frame | 🔲 Day 2 | src/reconstruction/zernike.py |
| 3 | r₀ — Fried parameter | 🔲 Day 3 | src/turbulence/estimators.py |
| 4 | τ₀ — Coherence time | 🔲 Day 3 | src/turbulence/estimators.py |
| 5 | A(xi,yi) — Actuator map | 🔲 Day 4 | src/actuator/dm_control.py |

## ⚖️ ISRO's 3 Evaluation Criteria

| # | Criterion | Target | Status |
|---|-----------|--------|--------|
| V1 | Wavefront accuracy | RMS < 50nm, Strehl > 0.70 | 🔲 |
| V2 | Turbulence parameters | r₀ error < 5%, τ₀ error < 10% | 🔲 |
| V3 | Speed | < 10ms per frame | 🔲 |

## 🗓️ Daily Workflow: Research → Validate → Implement → Replicate

Each day follows this loop:
1. **Research** (morning): understand the physics/algorithm
2. **Validate** (afternoon): answer quiz questions without notes
3. **Implement POC** (evening): working code in notebook
4. **Replicate** (same evening): clean version committed to src/

## 🚀 Quick Start

```bash
# Install dependencies
pip install hcipy aotools opencv-python torch scipy matplotlib astropy streamlit

# Run tests
python tests/test_simulator.py
python tests/test_metrics.py

# Generate one training pair (test Day 1 code)
python src/data/simulator.py

# Run full pipeline on ISRO data (after Day 5)
# python -c "from src.pipeline import PS09Pipeline; ..."
```

## 📊 Progress Tracker

| Module | Day | Status |
|--------|-----|--------|
| config.py | 1 | ✅ |
| src/data/simulator.py | 1 | ✅ |
| src/data/loader.py | 1 | ✅ |
| src/utils/metrics.py | 1 | ✅ |
| src/utils/visualise.py | 1 | ✅ |
| src/data/centroiding.py | 2 | 🔲 |
| src/reconstruction/zernike.py | 2 | 🔲 |
| src/reconstruction/classical.py | 3 | 🔲 |
| src/turbulence/estimators.py | 3 | 🔲 |
| src/models/isnet.py | 4 | 🔲 |
| src/models/lstm.py | 4 | 🔲 |
| src/actuator/dm_control.py | 4 | 🔲 |
| src/pipeline.py | 5 | 🔲 |
| demo/app.py | 5 | 🔲 |
