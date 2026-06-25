"""Assemble the HW3 Fourier deliverable into an organized, zipped package
with a self-contained HTML report (narrative + inline plots + stat tables)."""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
STAGE = ROOT / "_package" / "HW3_Fourier_Track"
if STAGE.parent.exists():
    shutil.rmtree(STAGE.parent)
(STAGE / "plots" / "data_eda").mkdir(parents=True)
(STAGE / "plots" / "fourier_model").mkdir(parents=True)
(STAGE / "plots" / "cluster_ablation").mkdir(parents=True)
(STAGE / "stats").mkdir()
(STAGE / "code").mkdir()

# ---- copy plots ----
eda_plots = ["01_class_distribution.png", "02_windows_per_subject.png",
             "03_example_windows.png", "04_mean_spectrum.png",
             "05_motion_std_hist.png", "06_axis_gravity_means.png"]
for p in eda_plots:
    shutil.copy(OUT / "eda" / p, STAGE / "plots" / "data_eda" / p)
for p in ["confusion_matrix.png", "spectra_per_activity.png", "feature_importance.png"]:
    shutil.copy(OUT / p, STAGE / "plots" / "fourier_model" / p)
for p in ["ablation_comparison.png", "confusion_matrix_best.png", "per_subject_f1.png"]:
    shutil.copy(OUT / "cluster" / p, STAGE / "plots" / "cluster_ablation" / p)

# ---- copy stats ----
shutil.copy(OUT / "eda" / "data_stats.json", STAGE / "stats" / "data_stats.json")
shutil.copy(OUT / "metrics.json", STAGE / "stats" / "fourier_model_metrics.json")
shutil.copy(OUT / "classification_report.txt", STAGE / "stats" / "fourier_model_report.txt")
shutil.copy(OUT / "cluster" / "summary.json", STAGE / "stats" / "cluster_ablation_summary.json")
shutil.copy(OUT / "cluster" / "best_classification_report.txt", STAGE / "stats" / "cluster_best_report.txt")

# ---- copy code ----
shutil.copy(ROOT / "fourier_model.py", STAGE / "code" / "fourier_model.py")
shutil.copy(ROOT / "eda.py", STAGE / "code" / "eda.py")
shutil.copy(ROOT / "plot_cluster.py", STAGE / "code" / "plot_cluster.py")
shutil.copy(ROOT.parent / "cluster" / "har_experiment.py", STAGE / "code" / "har_experiment.py")
shutil.copy(ROOT.parent / "cluster" / "run_har.slurm", STAGE / "code" / "run_har.slurm")

# ---- copy the markdown report ----
shutil.copy(ROOT / "REPORT.md", STAGE / "REPORT.md")

# ---- build a self-contained HTML report ----
stats = json.loads((OUT / "eda" / "data_stats.json").read_text())


def img(path, cap):
    return (f'<figure><img src="{path}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>HW3 — Fourier-Transform Track</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       max-width:1000px;margin:24px auto;padding:0 18px;color:#1c1c1e;line-height:1.55}}
 h1{{border-bottom:3px solid #2a9d8f;padding-bottom:6px}}
 h2{{margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:4px;color:#264653}}
 table{{border-collapse:collapse;margin:12px 0;font-size:14px}}
 th,td{{border:1px solid #ccc;padding:5px 10px;text-align:center}}
 th{{background:#f0f4f4}}
 figure{{margin:18px 0;text-align:center}}
 img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
 figcaption{{font-size:13px;color:#555;margin-top:6px}}
 code{{background:#f4f4f4;padding:1px 5px;border-radius:4px}}
 .key{{background:#e8f5f3;border-left:4px solid #2a9d8f;padding:10px 14px;margin:14px 0}}
</style></head><body>

<h1>HW3 — Fourier-Transform Track: Process Report</h1>
<p><b>Project:</b> Real-Time Human Activity Recognition from Smartphone Sensors.
<b>Scope:</b> our team owns the Fourier-transform modeling technique. This report
covers the full process: data understanding, a classical Fourier-feature model, a
Fourier+Transformer deep model, and a GPU-cluster robustness study.</p>

<h2>1. The data (UCI HAR)</h2>
<p>Smartphone accelerometer + gyroscope at 50 Hz, waist-worn, 6 activities, already
windowed into 128-sample (2.56 s) windows with 9 raw inertial channels. The
train/test split is <b>subject-disjoint</b>, so we measure generalization to new
people.</p>
<table>
<tr><th>Windows</th><th>Window</th><th>Classes</th><th>Balance ratio</th><th>Per-subject windows</th></tr>
<tr><td>7,352 train (21 subj) + 2,947 test (9 subj)</td><td>128 @ 50 Hz = 2.56 s, 9 ch</td>
<td>6</td><td>{stats['class_balance_ratio_train']}</td>
<td>{stats['per_subject_windows_train']['min']}–{stats['per_subject_windows_train']['max']} (mean {stats['per_subject_windows_train']['mean']})</td></tr>
</table>

<h3>Exploratory data analysis</h3>
{img("plots/data_eda/01_class_distribution.png", "Class distribution — train and test keep the same proportions; balanced.")}
{img("plots/data_eda/03_example_windows.png", "One window per activity. Walking variants oscillate; static activities are flat.")}
{img("plots/data_eda/05_motion_std_hist.png", "Within-window std of body-acc magnitude almost perfectly separates dynamic vs static.")}
{img("plots/data_eda/06_axis_gravity_means.png", "Per-axis gravity. SITTING vs STANDING differ only subtly (y/z) — the source of their confusion; LAYING flips onto y/z.")}
{img("plots/data_eda/04_mean_spectrum.png", "Average FFT spectrum per class: walking gait peak ~1.6 Hz; static near-zero. Justifies Fourier features.")}
{img("plots/data_eda/02_windows_per_subject.png", "Windows per training subject — evenly spread (281–409).")}

<div class="key"><b>Motion energy by class (within-window std, g):</b>
WALKING {stats['motion_std_by_class_g']['WALKING']},
WALK_UP {stats['motion_std_by_class_g']['WALK_UP']},
WALK_DOWN {stats['motion_std_by_class_g']['WALK_DOWN']},
SITTING {stats['motion_std_by_class_g']['SITTING']},
STANDING {stats['motion_std_by_class_g']['STANDING']},
LAYING {stats['motion_std_by_class_g']['LAYING']}.</div>

<h2>2. Classical Fourier-feature model</h2>
<p>We compute our own 165 frequency-domain features (11 channels × 15 spectral
features) from the raw windows via FFT, then classify with LogReg / RBF-SVM /
Random Forest (subject-aware GroupKFold selection).</p>
<table>
<tr><th>Model</th><th>Test accuracy</th><th>Macro-F1</th></tr>
<tr><td>Logistic Regression</td><td>0.925</td><td>0.926</td></tr>
<tr><td>RBF SVM</td><td>0.928</td><td>0.929</td></tr>
<tr><td><b>Random Forest</b></td><td><b>0.936</b></td><td><b>0.936</b></td></tr>
</table>
{img("plots/fourier_model/confusion_matrix.png", "Zero dynamic-vs-static confusion; only error is SITTING vs STANDING.")}
{img("plots/fourier_model/spectra_per_activity.png", "Per-activity FFT spectra used to build the features.")}
{img("plots/fourier_model/feature_importance.png", "Most informative Fourier features (random forest).")}

<h2>3. Fourier + Transformer deep model</h2>
<p>A dual-branch transformer: self-attention over the 128 time tokens AND over the
65 FFT bins (FFT done as a DFT matmul, so it runs on GPU/TPU without
<code>torch.fft</code>); the two CLS embeddings are fused before the classifier
(~148k parameters).</p>

<h2>4. The problem — cross-subject generalization</h2>
<p>Models fit the training subjects (train macro-F1 ≈ 0.99) but drop on unseen
people (test ≈ 0.93). The error is concentrated: dynamic↔static is never confused,
LAYING is near-perfect, and almost all error is SITTING↔STANDING plus a few hard
subjects (9, 10).</p>

<h2>5. Robustness study on the GPU cluster</h2>
<p>Seed-averaged ablation (3 seeds × 3 configs × 25 epochs) on one GPU (GTX 1080
Ti), augmenting the raw 128×9 windows in training: jitter + scaling + small
rotation (±20°, deliberately small so it doesn't erase the gravity cue).</p>
<table>
<tr><th>Config</th><th>Test macro-F1</th><th>Test acc</th><th>Overfit gap</th></tr>
<tr><td>baseline</td><td>0.9287 ± 0.0046</td><td>0.931</td><td>+0.063</td></tr>
<tr><td><b>aug (jitter+scale+rotate)</b></td><td><b>0.9459 ± 0.0013</b></td><td><b>0.947</b></td><td><b>+0.029</b></td></tr>
<tr><td>aug + heavy reg</td><td>0.9122 ± 0.0178</td><td>0.913</td><td>+0.046</td></tr>
</table>
{img("plots/cluster_ablation/ablation_comparison.png", "Augmentation alone gives the best test F1, the smallest overfit gap, and the lowest seed variance. Heavy regularization on top hurts.")}
{img("plots/cluster_ablation/confusion_matrix_best.png", "Best model (aug) confusion matrix — residual error still SITTING/STANDING, now softened.")}
{img("plots/cluster_ablation/per_subject_f1.png", "Per-subject generalization: subjects 9 and 10 are hardest (red), confirming the diagnosis.")}

<div class="key"><b>Key finding:</b> augmentation is the single most effective tool
for cross-subject robustness — +1.7 macro-F1 points, overfit gap more than halved,
lowest variance. Stacking heavy regularization on top was counterproductive at this
epoch budget.</div>

<h2>6. Next steps</h2>
<ul>
<li>Add a small tilt/orientation feature to attack SITTING↔STANDING directly.</li>
<li>Run true Leave-One-Subject-Out CV to quantify the hard subjects.</li>
<li>For the live Phyphox demo, raise rotation strength to trade UCI accuracy for
cross-orientation transfer.</li>
</ul>

<hr><p style="font-size:13px;color:#777">Reproducible: local <code>conda activate study</code>;
cluster <code>conda activate subpop</code> (GPU). See <code>code/</code> and
<code>stats/</code>.</p>
</body></html>"""

(STAGE / "index.html").write_text(html)

# ---- zip ----
zip_path = ROOT.parent / "HW3_Fourier_Track.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(STAGE.rglob("*")):
        if f.is_file():
            zf.write(f, f.relative_to(STAGE.parent))

total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
print(f"Package staged at {STAGE}")
print(f"ZIP written: {zip_path} ({zip_path.stat().st_size/1024:.0f} KB, "
      f"{sum(1 for f in STAGE.rglob('*') if f.is_file())} files)")
