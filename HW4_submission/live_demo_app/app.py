"""
Live HAR demo — Fourier+Transformer (deep model), three-state real-time view.

Receives the phone's Sensor Logger stream, windows it (2.56 s @ 50 Hz), runs the
trained FourierTransformer (fourier_transformer.py + checkpoint in artifacts/),
and shows the predicted activity live as one of three steady states:
walking / sitting / lying.

Run:
    ../.venv/bin/python app.py
    # then point Sensor Logger (HTTP push) at  http://<this-machine-LAN-IP>:5050/data
"""
import json
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from fourier_transformer import load_model, predict
from live_core import (autoscale_to_g, resample_window, build_window,
                       collapse_probs, DEMO_LABELS as LABELS,
                       DEMO_EMOJI as EMOJI, WINDOW_SEC, FS, N)

app = Flask(__name__)
CORS(app)

HERE = Path(__file__).resolve().parent
MODEL, META = load_model(HERE / "artifacts" / "fourier_transformer_best.pt",
                         HERE / "artifacts" / "model_meta.json", device="cpu")
MODEL_KIND = f"fourier_transformer (deep, test acc {META.get('test_acc'):.3f})"

# ---------------------------------------------------------------- state
BUF = 600  # ~12 s at 50 Hz
_lock = threading.Lock()
_accel = deque(maxlen=BUF)   # (t, x, y, z)  total acceleration
_gyro = deque(maxlen=BUF)    # (t, x, y, z)  angular velocity
_counts = {"accelerometer": 0, "gyroscope": 0, "other": 0, "posts": 0}
_last_post_wall = None

_ema = None
_EMA_ALPHA = 0.55
_pred = {"activity": None, "confidence": 0.0,
         "probs": {l: 0.0 for l in LABELS}, "status": "waiting for data"}
_last_pred_wall = 0.0
_PRED_MIN_GAP = 0.25
MIN_SAMPLES = 40

ACCEL_NAMES = {"accelerometer", "accelerometeruncalibrated"}
GYRO_NAMES = {"gyroscope", "gyroscopeuncalibrated"}


def _ingest(body):
    n = 0

    def add(name, t, vals):
        nonlocal n
        if not isinstance(vals, dict) or t is None:
            return
        x, y, z = vals.get("x"), vals.get("y"), vals.get("z")
        if x is None:
            return
        if name in ACCEL_NAMES:
            _accel.append((t, x, y, z)); _counts["accelerometer"] += 1; n += 1
        elif name in GYRO_NAMES:
            _gyro.append((t, x, y, z)); _counts["gyroscope"] += 1; n += 1
        else:
            _counts["other"] += 1

    if isinstance(body, dict) and isinstance(body.get("payload"), list):
        for e in body["payload"]:
            if isinstance(e, dict):
                t = e.get("time")
                t = (t / 1e9) if isinstance(t, (int, float)) else None
                add(str(e.get("name", "")).lower(), t, e.get("values", {}))
    elif isinstance(body, list):
        for e in body:
            if isinstance(e, dict):
                add(str(e.get("sensor", "accelerometer")).lower(), e.get("t"), e)
    return n


def _predict_locked():
    global _ema, _pred, _last_pred_wall
    if time.time() - _last_pred_wall < _PRED_MIN_GAP:
        return
    if len(_accel) < MIN_SAMPLES or len(_gyro) < MIN_SAMPLES:
        _pred["status"] = "collecting… (move the phone; raise the sample rate)"
        return

    ta = np.array([s[0] for s in _accel]); va = np.array([s[1:] for s in _accel])
    tg = np.array([s[0] for s in _gyro]);  vg = np.array([s[1:] for s in _gyro])
    t_end = min(ta[-1], tg[-1])
    if ((ta >= t_end - WINDOW_SEC) & (ta <= t_end)).sum() < MIN_SAMPLES:
        _pred["status"] = "collecting…"
        return

    acc = autoscale_to_g(resample_window(ta, va, t_end))   # (128,3) g  -> total_acc
    gyr = resample_window(tg, vg, t_end)                   # (128,3) rad/s
    raw = build_window(acc, gyr)                           # (1,128,9) RAW
    _, _, probs6 = predict(MODEL, raw, META, device="cpu")  # their code z-scores inside
    p = collapse_probs(probs6[0])                          # (3,) walking/sitting/lying
    _ema = p if _ema is None else (_EMA_ALPHA * p + (1 - _EMA_ALPHA) * _ema)

    top = int(np.argmax(_ema))
    _pred = {"activity": LABELS[top], "confidence": float(_ema[top]),
             "probs": {LABELS[i]: float(_ema[i]) for i in range(len(LABELS))},
             "status": "ok"}
    _last_pred_wall = time.time()


@app.post("/data")
def data():
    global _last_post_wall
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify(ok=False, error="no JSON body"), 400
    with _lock:
        _counts["posts"] += 1
        _last_post_wall = time.time()
        got = _ingest(body)
        _predict_locked()
    return jsonify(ok=True, received=got, activity=_pred["activity"])


def _effective_hz():
    with _lock:
        ts = [s[0] for s in _accel]
    if len(ts) < 2:
        return 0.0
    span = ts[-1] - ts[0]
    return (len(ts) - 1) / span if span > 0 else 0.0


@app.get("/state")
def state():
    with _lock:
        counts = dict(_counts); pred = json.loads(json.dumps(_pred))
        last_post = _last_post_wall
        a = _accel[-1] if _accel else None
        stream = [{"x": round(s[1], 3), "y": round(s[2], 3), "z": round(s[3], 3)}
                  for s in list(_accel)[-160:]]
    age = (time.time() - last_post) if last_post else None
    return jsonify(
        connected=(age is not None and age < 3.0),
        seconds_since_last_post=(round(age, 1) if age is not None else None),
        hz=round(_effective_hz(), 1),
        counts=counts, prediction=pred,
        emoji={l: EMOJI[l] for l in LABELS},
        last_accel=(None if a is None else {"x": a[1], "y": a[2], "z": a[3]}),
        stream=stream,
    )


@app.get("/health")
def health():
    return jsonify(ok=True, counts=_counts, hz=round(_effective_hz(), 1),
                   states=LABELS,
                   model={"model_kind": MODEL_KIND, "test_acc": META.get("test_acc"),
                          "architecture": META.get("architecture")})


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>What am I doing? (Transformer)</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0d1326;color:#e8eefc;margin:0;padding:22px;text-align:center}
 .sub{color:#9fb4d8;font-size:13px;margin-bottom:14px}
 #hero{background:#16224a;border:1px solid #283a78;border-radius:18px;padding:26px;margin:0 auto 16px;max-width:520px}
 #emoji{font-size:84px;line-height:1} #act{font-size:30px;font-weight:800;margin-top:4px;letter-spacing:.5px}
 #conf{color:#9fb4d8;margin-top:6px;font-size:14px} #status{color:#ffd479;font-size:13px;min-height:16px;margin-top:6px}
 .bars{max-width:520px;margin:0 auto 16px;text-align:left}
 .row{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13px}
 .row .lbl{width:160px;color:#cdd9f2} .track{flex:1;background:#1b274f;border-radius:6px;height:16px;overflow:hidden}
 .fill{height:100%;background:linear-gradient(90deg,#7b5cff,#34d27b);width:0%}
 .meta{color:#7e90b8;font-size:12px} .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
 #live{max-width:520px;margin:14px auto 8px;background:#10193a;border:1px solid #283a78;border-radius:14px;padding:12px}
 #live .hd{display:flex;justify-content:space-between;font-size:12px;color:#9fb4d8;margin-bottom:6px}
 @keyframes pulse{0%{opacity:1}50%{opacity:.25}100%{opacity:1}} .pulsing{animation:pulse 1s infinite}
 canvas{width:100%;height:140px;display:block;background:#0a1230;border-radius:8px}
</style></head><body>
 <div class="sub">🧠 live activity recognition — Fourier+Transformer (deep model) on your phone's sensors</div>
 <div id="hero"><div id="emoji">⏳</div><div id="act">…</div><div id="conf"></div><div id="status"></div></div>
 <div id="live"><div class="hd"><div id="recv">⚪ waiting for stream…</div>
   <div>accel <span style="color:#ff6b6b">x</span><span style="color:#51cf66">y</span><span style="color:#4dabf7">z</span></div></div>
   <canvas id="chart" width="900" height="280"></canvas></div>
 <div class="bars" id="bars"></div>
 <div class="meta"><span id="conn"></span> · <span id="hz">0</span> Hz · <span id="cnt">0</span> samples</div>
<script>
 const pct=(x)=>(x*100).toFixed(0);
 const cv=document.getElementById('chart'),cx=cv.getContext('2d');
 function drawChart(stream){const W=cv.width,H=cv.height;cx.clearRect(0,0,W,H);
   cx.strokeStyle='#1b2750';cx.lineWidth=1;for(let i=0;i<=4;i++){const y=H*i/4;cx.beginPath();cx.moveTo(0,y);cx.lineTo(W,y);cx.stroke();}
   if(!stream||stream.length<2)return;let m=0.5;stream.forEach(s=>{m=Math.max(m,Math.abs(s.x),Math.abs(s.y),Math.abs(s.z));});m*=1.15;
   const mid=H/2,sc=(H/2)/m,n=stream.length;[['x','#ff6b6b'],['y','#51cf66'],['z','#4dabf7']].forEach(([k,c])=>{
     cx.strokeStyle=c;cx.lineWidth=1.6;cx.beginPath();stream.forEach((s,i)=>{const X=W*i/(n-1),Y=mid-s[k]*sc;i?cx.lineTo(X,Y):cx.moveTo(X,Y);});cx.stroke();});}
 let lastCnt=0;
 async function tick(){try{
   const s=await (await fetch('/state')).json();const p=s.prediction,on=s.connected;
   document.getElementById('emoji').textContent=p.activity?s.emoji[p.activity]:'⏳';
   document.getElementById('act').textContent=p.activity?p.activity.replace(/_/g,' '):'…';
   document.getElementById('conf').textContent=p.activity?('confidence '+pct(p.confidence)+'%'):'';
   document.getElementById('status').textContent=p.status==='ok'?'':p.status;
   const order=Object.keys(s.emoji);
   document.getElementById('bars').innerHTML=order.map(l=>{const v=p.probs[l]||0;
     return '<div class="row"><div class="lbl">'+s.emoji[l]+' '+l.replace(/_/g,' ')+'</div><div class="track"><div class="fill" style="width:'+pct(v)+'%"></div></div><div style="width:38px">'+pct(v)+'%</div></div>';}).join('');
   document.getElementById('conn').innerHTML='<span class="dot" style="background:'+(on?'#34d27b':'#e0556b')+'"></span>'+(on?'connected':'waiting');
   document.getElementById('hz').textContent=s.hz;document.getElementById('cnt').textContent=s.counts.accelerometer;
   const growing=s.counts.accelerometer>lastCnt;lastCnt=s.counts.accelerometer;const recv=document.getElementById('recv');
   if(growing){recv.innerHTML='🟢 receiving — '+s.counts.accelerometer+' samples @ '+s.hz+' Hz';recv.className='pulsing';}
   else if(on){recv.innerHTML='🟡 connected, idle';recv.className='';}else{recv.innerHTML='⚪ waiting for stream…';recv.className='';}
   drawChart(s.stream);
  }catch(e){document.getElementById('status').textContent='server error';}}
 setInterval(tick,200);tick();
</script></body></html>"""


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    ip = _lan_ip()
    print("=" * 64)
    print(f" Live HAR server — {MODEL_KIND}")
    print(f"   States    : {', '.join(LABELS)}")
    print(f"   Dashboard : http://localhost:{port}/   |   http://{ip}:{port}/")
    print(f"   POST      : http://{ip}:{port}/data")
    print("=" * 64, flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
