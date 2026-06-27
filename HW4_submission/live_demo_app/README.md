# Live HAR demo (real-time activity recognition)

Streams a phone's accelerometer + gyroscope to the trained **Fourier+Transformer** and shows the
predicted activity live in the browser. For the workshop this runs **in the cloud on a RunPod pod**,
from the Docker image on Docker Hub — **not on the laptop**.

## Run it (RunPod, from Docker Hub)

1. **Deploy a RunPod pod** with the published image and expose the HTTP port:
   - Container image: the Docker Hub tag (`korkevados/har-live-transformer:latest`)
   - Expose HTTP port: `5050`  ·  Env var `PORT=5050`  ·  leave the start command empty
   (full walkthrough in `../../docker_image/DEPLOY_RUNPOD.md`).
2. When the pod is **Running**, open **Connect → HTTP :5050** → `https://<podid>-5050.proxy.runpod.net/`.
   (First request takes a few seconds while torch + the checkpoint load.)
3. On the phone, install **Sensor Logger**, enable *Accelerometer* + *Gyroscope*, turn on **HTTP push**,
   and set the push URL to `https://<podid>-5050.proxy.runpod.net/data` (raise the rate to ~50 Hz).
4. Start streaming and move the phone — the emoji and bars update in real time.

## What it shows

Three steady real-time states: **walking · sitting · lying**. An exponential moving average smooths
the live output. Inference is a few milliseconds per window, so the prediction tracks movement closely.

## Files (the service that the image is built from)

- `app.py` — Flask server + live dashboard (HTML).
- `live_core.py` — windowing/resampling of the phone stream + class grouping.
- `fourier_transformer.py` — model architecture + load/predict.
- `artifacts/` — trained checkpoint, metadata, and a few sample windows.

## Optional: local smoke test

You can run the service locally just to check it works (not how the demo is presented):

```bash
../.venv/bin/python app.py   # then open the printed http://localhost:5050/
```
