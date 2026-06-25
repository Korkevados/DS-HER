# HW3 — Fourier-Transform Track

Our team's modeling technique for the **Real-Time Human Activity Recognition**
project: classify a 2.56 s smartphone-sensor window into one of six activities
using **Fourier (frequency-domain) features**. Two models are delivered — a
classical Fourier-feature classifier and a Fourier+Transformer deep model — plus
a full data EDA and a GPU-cluster robustness study.

## Results (UCI HAR test set, subject-disjoint)

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Fourier features + Logistic Regression | 0.925 | 0.926 |
| Fourier features + Random Forest | 0.936 | 0.936 |
| **Fourier + Transformer (with augmentation)** | **0.948** | **0.946** |

Augmentation (jitter + scaling + small rotation) is the key lever for
cross-subject generalization: it raised test macro-F1 by ~1.7 points and **halved
the train→test overfitting gap** (0.063 → 0.029).

## Layout

```
hw3_fourier/
├── README.md                     this file
├── REPORT.md                     full process report (data → models → cluster study)
├── HW3_modeling_fourier.md       CRISP-DM template text, sections 4.1–4.4
├── fourier_model.py              classical Fourier-feature model (train + evaluate + save)
├── eda.py                        data exploratory analysis (plots + stats)
├── plot_cluster.py               renders ablation plots from the cluster summary
├── assemble_zip.py               builds the report deliverable zip
├── models_package/               self-contained, runnable models bundle
│   ├── USAGE.md                  how to use the models (inputs/outputs/preprocessing)
│   ├── code/                     loaders + predict.py demo
│   ├── models/                   trained weights (.pt, .joblib) + model_meta.json
│   └── sample/                   12 real test windows for the demo
└── outputs/                      generated metrics, plots, and trained artifacts
    ├── *.png, metrics.json, classification_report.txt
    ├── fourier_model_best.joblib
    ├── eda/                      6 data EDA plots + data_stats.json
    └── cluster/                  ablation plots, summary.json, best model .pt

../cluster/                       GPU-cluster experiment
├── har_experiment.py             Fourier+Transformer robustness ablation
└── run_har.slurm                 SLURM job script
```

## Reproduce

```bash
# data: place the UCI HAR Dataset under ../data/  (gitignored)
conda activate study           # local: numpy, pandas, scikit-learn, scipy, matplotlib

python fourier_model.py        # classical model -> outputs/
python eda.py                  # data EDA        -> outputs/eda/

# deep model on a GPU (cluster): conda activate <gpu-env>
python ../cluster/har_experiment.py --data_dir "../data/UCI HAR Dataset" --out_dir results
```

## Use the trained models

See [models_package/USAGE.md](models_package/USAGE.md). Quick start:

```bash
cd models_package/code && python predict.py     # runs both models on the bundled sample
```

Input to both models: raw windows `(N, 128, 9)` at 50 Hz (channels:
`total_acc xyz`, `body_acc xyz`, `body_gyro xyz`; accel in g, gyro in rad/s).
Output: an activity label per window (+ probabilities for the deep model).
Preprocessing (normalization / feature extraction) is handled inside the code.
