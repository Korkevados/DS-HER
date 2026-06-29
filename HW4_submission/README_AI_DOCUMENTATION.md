# DS-HER Project: AI-Readable Documentation Suite

## Overview

This directory contains comprehensive technical documentation for the **DS-HER (Human Activity Recognition)** deep learning project, specifically optimized for AI systems to understand the entire project architecture, algorithms, results, and implementation details.

## Team

- **Almog Tal**
- **Daniel Korkevados**  
- **Lior Sulshtein**
- **Ilay Damari**

**Institution:** Ben-Gurion University of the Negev

## Documentation Files

### 1. **AI_COMPREHENSIVE_DOCUMENTATION.md** (Primary Document)
   - **Purpose:** Complete project overview for AI systems
   - **Scope:** 87,000+ words, 12 major sections
   - **Coverage:**
     - Executive summary & problem statement
     - Exploratory data analysis (EDA) with key findings
     - All model architectures (8 variants)
     - Complete benchmark results
     - Deployment architecture & specifications
     - Evaluation methodology
     - Reproducibility guide
     - Limitations & future work
     - References & resources
     - Technical appendices

### 2. **TECHNICAL_SPECIFICATIONS.md** (Implementation Details)
   - **Purpose:** Code-level implementation guide
   - **Scope:** Algorithms, data structures, API specs
   - **Coverage:**
     - Data pipeline specifications
     - Complete PyTorch model implementations
     - Training loop pseudocode
     - Flask deployment API
     - Docker containerization
     - Performance benchmarks
     - Optimization opportunities
     - On-device deployment guides

### 3. **HW4_submission/** (Final Deliverable)
   - **Location:** `/HW4_submission/`
   - **Contents:**
     - Pre-executed Jupyter notebook (`HAR_HW4_display.ipynb`)
     - PowerPoint presentation (`HAR_HW4_display.pptx`)
     - Live demo application (Flask + Docker)
     - Trained model artifacts
     - All experimental results (JSON format)

## Quick Reference

### Project Stats

| Metric | Value |
|--------|-------|
| **Best Model** | Fourier+Transformer (dual-branch) |
| **Test Accuracy** | 94.8% |
| **Test Macro-F1** | 94.8% |
| **Parameters** | 130,374 (lightweight) |
| **Inference Latency** | ~21 ms (CPU) |
| **Dataset** | UCI HAR (7,352 train / 2,947 test windows) |
| **Task** | 6-class activity recognition |
| **Deployment** | Docker + Flask on RunPod cloud |

### Key Innovation

**Dual-Branch Architecture:**
- **Time Branch:** Self-attention over 128 raw timesteps
- **Frequency Branch:** Self-attention over 65 FFT magnitude bins
- **Late Fusion:** Concatenate CLS representations → classifier
- **Result:** Resolves SITTING ↔ STANDING ambiguity that plagued single-branch models

### Model Comparison (Test Macro-F1)

```
1. LogReg (561 engineered features)     95.6%  ← Classical SOTA
2. Fourier+Transformer (augmented)      94.8%  ← DEPLOYED ✓
3. SVM-RBF (561 features)               95.3%
4. Random Forest (165 Fourier features) 93.6%
5. Patch Transformer                    89.9%
6. CNN-Transformer Hybrid               89.6%
7. Main Transformer                     88.2%
8. LogReg (6 selected features)         78.5%
```

## Repository Structure

```
DS-HER/
├── AI_COMPREHENSIVE_DOCUMENTATION.md    ← Main AI-readable doc
├── TECHNICAL_SPECIFICATIONS.md          ← Implementation details
├── README_AI_DOCUMENTATION.md           ← This file
├── HW4_submission/                      ← Final deliverable
│   ├── option1_display_only/            ← Notebook + slides
│   │   ├── HAR_HW4_display.ipynb
│   │   ├── HAR_HW4_display.pptx
│   │   ├── fourier_transformer.py       ← Model definition
│   │   ├── har_utils.py                 ← Data loading
│   │   ├── har_plots.py                 ← Visualization
│   │   ├── metrics.json                 ← All results
│   │   └── artifacts/
│   │       ├── fourier_transformer_best.pt  ← Checkpoint
│   │       └── model_meta.json          ← Metadata
│   └── live_demo_app/                   ← Real-time deployment
│       ├── app.py                       ← Flask server
│       ├── fourier_transformer.py       ← Model loader
│       ├── live_core.py                 ← Preprocessing
│       ├── Dockerfile                   ← Container image
│       └── artifacts/                   ← (same as above)
├── transformer + fouria/                ← Full training pipeline
│   ├── har_full_run.py                  ← Complete experiment script
│   └── README.md                        ← SLURM cluster guide
├── baselines/                           ← Classical ML experiments
│   ├── hw3_modeling.py                  ← 561-feature baselines
│   └── select6_features.py              ← Feature selection
├── hw3_fourier/                         ← Fourier feature experiments
│   └── fourier_model.py                 ← 165 spectral features
└── docs filled/                         ← Assignment submissions (PDFs)
```

## How AI Systems Should Use This Documentation

### For Understanding the Project

1. **Start with:** `AI_COMPREHENSIVE_DOCUMENTATION.md` § 1-2
   - Read Executive Summary
   - Understand problem statement & dataset

2. **Deep Dive:** `AI_COMPREHENSIVE_DOCUMENTATION.md` § 3-5
   - Study all model architectures
   - Review benchmark results
   - Analyze deployment architecture

3. **Code-Level:** `TECHNICAL_SPECIFICATIONS.md`
   - Review PyTorch implementations
   - Understand training procedures
   - Study API specifications

### For Reproducing Results

1. **Environment Setup:**
   ```bash
   cd HW4_submission
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Quick Demo (Pre-executed):**
   ```bash
   jupyter notebook option1_display_only/HAR_HW4_display.ipynb
   # Notebook is already executed - just scroll through
   ```

3. **Full Training (Reproduce from scratch):**
   ```bash
   cd "transformer + fouria"
   python har_full_run.py --data /path/to/"UCI HAR Dataset"
   # Outputs: metrics.json, training curves, confusion matrices
   ```

4. **Live Deployment:**
   ```bash
   cd HW4_submission/live_demo_app
   docker build -t har-live .
   docker run -p 5050:5050 har-live
   # Open: http://localhost:5050/
   ```

### For Extending the Work

Refer to `AI_COMPREHENSIVE_DOCUMENTATION.md` § 9.2 for:
- Architecture improvements (Conformer, TCN, self-supervised pre-training)
- Dataset extensions (cross-dataset evaluation, transition detection)
- Deployment optimizations (quantization, distillation, on-device inference)

## Key Files for AI Analysis

### Model Architecture
- **Primary:** `HW4_submission/option1_display_only/fourier_transformer.py`
- **Reference:** `TECHNICAL_SPECIFICATIONS.md` § 2.1 (full PyTorch code)

### Training Code
- **Full Pipeline:** `transformer + fouria/har_full_run.py` (self-contained, 724 lines)
- **Pseudocode:** `TECHNICAL_SPECIFICATIONS.md` § 5.2

### Results & Metrics
- **All Experiments:** `HW4_submission/option1_display_only/metrics.json`
- **Model Metadata:** `HW4_submission/option1_display_only/artifacts/model_meta.json`
- **Analysis:** `AI_COMPREHENSIVE_DOCUMENTATION.md` § 4 & Appendix B

### Deployment
- **Flask Server:** `HW4_submission/live_demo_app/app.py`
- **Dockerfile:** `HW4_submission/live_demo_app/Dockerfile`
- **API Spec:** `TECHNICAL_SPECIFICATIONS.md` § 3.1

## Critical Insights for AI Systems

### 1. Subject-Disjoint Evaluation is Essential
- **Never** randomly shuffle windows for train/test split
- UCI HAR has 50% window overlap → adjacent windows are correlated
- **Always** keep all windows from a subject together (train OR test, never both)
- Subject-disjoint F1 is ~5% lower than random-split (honest evaluation)

### 2. Frequency Domain is Key
- Walking activities have sharp spectral peak at **~1.6 Hz** (gait cadence)
- Static activities concentrate energy near **DC (0 Hz)**
- **FFT bin 0** captures per-axis gravity orientation → separates SITTING/STANDING
- Single time-domain Transformer fails at static posture discrimination

### 3. Dual-Branch Architecture Wins
- **Time branch:** Local temporal dependencies, gait phase
- **Frequency branch:** Spectral peaks, DC gravity signature
- **Fusion:** Complementary representations → 94.8% F1 (vs. 88.2% single-branch)

### 4. Data Augmentation is Critical
- 90.2% F1 (no augmentation) → 94.8% F1 (with augmentation) = **+4.6% gain**
- Effective: Gaussian noise, time shifting, amplitude scaling
- Ineffective: Rotation (breaks gravity orientation cues)

### 5. Classical Baselines Are Strong
- Logistic Regression on 561 engineered features: **95.6% F1**
- Deep learning (94.8%) is competitive but **not superior**
- Advantage of deep learning: End-to-end (no manual feature engineering), transferable

## Contact & Resources

- **Docker Image:** `korkevados/har-live-transformer:latest`
- **Dataset:** [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Framework:** PyTorch 2.0.1
- **Deployment:** RunPod cloud (Docker + Flask)

## Document Versions

- **AI_COMPREHENSIVE_DOCUMENTATION.md:** v1.0 (2024)
- **TECHNICAL_SPECIFICATIONS.md:** v1.0 (2024)
- **README_AI_DOCUMENTATION.md:** v1.0 (2024)

---

**For AI systems:** These documents are designed to be parsed and understood by large language models and AI research assistants. All technical details, algorithms, and results are explicitly documented with minimal ambiguity.

**For Humans:** Start with `HW4_submission/README.md` for a user-friendly guide to the workshop demo. The AI documentation above provides exhaustive technical depth.
