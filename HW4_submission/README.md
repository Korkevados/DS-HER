# HW4 — Real-Time Human Activity Recognition (Evaluation & Deployment)

Final CRISP-DM deliverable: a presentation + notebook workshop, a written report, and a
closing real-time demo. Everything runs **offline on a Mac/CPU** — no GPU, no training on the day.

**Students:** Almog Tal · Daniel Korkevados · Lior Sulshtein · Ilay Damari
**Model:** dual-branch Fourier+Transformer — **94.8% accuracy** on unseen subjects (UCI HAR).

---

## What's here

```
HW4_submission/
├── HW4_Final_Report.docx        # the written report (CRISP-DM Evaluation 5.x + Deployment 6.x)
├── option1_display_only/        # the presentation notebook + slides
│   ├── HAR_HW4_display.ipynb     #   pre-executed — every figure/result is already visible
│   └── HAR_HW4_display.pptx      #   the deck (no code on slides; cues to jump into the notebook)
├── live_demo_app/               # the closing live demo (phone → real-time prediction)
├── assets/                      # figures used by the slides
├── requirements.txt
└── .venv/                       # ready-to-use Python environment (created by setup below)
```

The notebook is **pre-run**, so during the talk you simply scroll to the section the current
slide points to (sections 1–8 map 1:1 to the slides). You jump slide → notebook → slide.

---

## One-time setup (already done on this machine)

```bash
cd HW4_submission
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name har-hw4 --display-name "Python (HAR HW4)"
```

The UCI HAR dataset is read from `../Data/human+activity+recognition+using+smartphones/`
(already in the project) — nothing to download.

---

## Running each piece

**Open the notebook**

```bash
.venv/bin/jupyter notebook option1_display_only/HAR_HW4_display.ipynb
```
It is already executed — just scroll. (To re-run, select the **Python (HAR HW4)** kernel; the
whole notebook finishes in seconds on CPU.)

**Open the deck** — double-click `option1_display_only/HAR_HW4_display.pptx`
(PowerPoint / Keynote / Google Slides).

**Run the closing live demo (on RunPod, not the laptop)**

The demo runs the Docker image pulled from Docker Hub on a **RunPod GPU pod** — not locally.
Deploy the image on a RunPod pod (expose port `5050`), open the pod's public **HTTPS** dashboard
URL, and point a phone sensor-streaming app (Sensor Logger, HTTP push) at `<pod-url>/data`, then
move the phone. Full steps in `live_demo_app/README.md` (and `docker_image/DEPLOY_RUNPOD.md`).

---

## Suggested workshop flow (avoids the graded pitfalls)

1. Title → motivation → research question → data **(slides)**
2. **↪ jump to notebook section 2** — walk the EDA graphs
3. Two EDA takeaways → modelling approach **(slides)**
4. **↪ jump to notebook section 3** — show the model architecture code
5. Test protocol → results **(slides)** with **↪ notebook sections 5–6** for the inference output
6. Ablations → evaluation vs criteria → deployment (Docker + inference) **(slides)**
7. Conclusions **(slide)** → **LIVE DEMO** → References

Slides never show code; the code lives in the notebook. That interleaving is deliberate.
