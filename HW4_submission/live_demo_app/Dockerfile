# Live HAR demo (real-time, 3-state view). Built for linux/amd64 (RunPod).
# CPU-only torch keeps the image small; the model is tiny so a GPU is not required.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# CPU-only torch from the official PyTorch index (no CUDA -> much smaller image).
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + model loader + checkpoint/meta/samples (in artifacts/).
COPY . .

ENV PORT=5050
EXPOSE 5050

# ONE worker: the live stream/prediction state is in-memory. Threads handle concurrency.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT:-5050} app:app"]
