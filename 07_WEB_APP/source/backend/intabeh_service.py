"""INTABEH — local inference service (Day 5; D-024).

Serves the Arabic web app (07_WEB_APP/public) and runs the REAL frozen model package
05_AI_MODEL/03_MODELS/FINAL_v1.0 (verified against FREEZE.json at startup, loaded once).

  GET  /health   -> service/model status and model version
  POST /predict  -> body = one audio chunk as a WAV file (any common rate >= 16 kHz, mono/stereo, <= 10 s)
                    -> D-014 preprocessing (channel mean, librosa soxr_hq -> 16 kHz, edge trim < 0.001,
                       clipping guard, centred pad to 15,600) -> frozen label-free inference v1.0
                       (-60 dBFS window gate, YAMNet, classifier, mean of windows) -> frozen thresholds policy
                    LIVE-001 / D-025: for live microphone audio the frozen -60 dBFS gate is evaluated on the band
                    YAMNet analyses (>= 125 Hz, its log-mel lower edge); YAMNet input, model and thresholds unchanged.

Privacy: audio is processed in memory only; nothing is written to disk or logged (except the opt-in LIVE-001
diagnostic recorder, OFF unless INTABEH_DIAG_DIR is set).
Run: .venv\\Scripts\\python.exe -m uvicorn intabeh_service:app --app-dir 07_WEB_APP/source/backend --port 8000
"""

import hashlib
import io
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "05_AI_MODEL" / "03_MODELS" / "FINAL_v1.0"
PUBLIC = ROOT / "07_WEB_APP" / "public"
# The frozen module derives a default cache path from its own location, which is only correct in
# 05_AI_MODEL/02_YAMNET; set the approved local YAMNet cache explicitly (no download at runtime).
os.environ.setdefault("TFHUB_CACHE_DIR", str(ROOT / ".venv" / "tfhub_cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True  # never write __pycache__ into the frozen, read-only package
sys.path.insert(0, str(PACKAGE))  # the frozen copy of the inference module is used
import intabeh_inference as inf  # noqa: E402

MODEL_VERSION = "FINAL_v1.0"
MAX_BYTES = 4 * 1024 * 1024
MAX_SECONDS, MIN_SECONDS = 10.0, 0.30
# User-facing Arabic labels (Day 5 UI specification). Internal class ids stay unchanged.
LABELS_AR = {"door_knock": "طرق الباب", "doorbell": "جرس الباب", "baby_cry": "بكاء طفل", "timer_alarm": "مؤقت / منبه",
             "phone_ring": "رنين الهاتف", "car_horn": "بوق السيارة", "dog_bark": "نباح كلب", "running_water": "ماء جارٍ",
             "smoke_fire_alarm": "صوت إنذار", "background_other": "صوت آخر / خلفية"}
ERRORS = {"empty": "لم يصل أي صوت. حاول مرة أخرى.",
          "too_large": "المقطع الصوتي أكبر من المسموح.",
          "unreadable": "تعذّر قراءة المقطع الصوتي.",
          "bad_rate": "معدّل العيّنات غير مدعوم (يجب ألا يقل عن 16 كيلوهرتز).",
          "too_long": "المقطع الصوتي أطول من المسموح (10 ثوانٍ).",
          "too_short": "المقطع الصوتي قصير جدًا.",
          "model_unavailable": "نموذج التعرّف غير جاهز بعد.",
          "inference_failed": "حدث خطأ أثناء تحليل الصوت."}
state = {"model": None, "cfg": None, "freeze_ok": False, "error": None}
# LIVE-001 (D-025): YAMNet's log-mel front end covers 125-7500 Hz. Laptop microphones add low-frequency rumble
# (< 125 Hz, e.g. fan / self-noise) that YAMNet does not hear but that lifts BROADBAND RMS above the frozen
# -60 dBFS gate, so near-silent windows reached the classifier, which maps near-silence to phone_ring (p ~ 1.0).
# The gate threshold is unchanged; it is measured on the >= 125 Hz band (a measurement copy; YAMNet input untouched).
GATE_BAND_HZ = 125.0
GATE_SOS = butter(4, GATE_BAND_HZ, btype="highpass", fs=16000, output="sos")
lock = threading.Lock()


def verify_freeze():
    rec = json.loads((PACKAGE / "FREEZE.json").read_text(encoding="utf-8"))
    return all(hashlib.sha256((PACKAGE / n).read_bytes()).hexdigest() == h for n, h in rec["files_sha256"].items())


@asynccontextmanager
async def lifespan(_app):
    try:
        state["freeze_ok"] = verify_freeze()
        if not state["freeze_ok"]:
            raise RuntimeError("FINAL_v1.0 files do not match FREEZE.json")
        state["cfg"] = json.loads((PACKAGE / "preprocessing_config_v1.0.json").read_text(encoding="utf-8"))
        state["model"] = inf.IntabehModel(PACKAGE)
        state["model"].clip_probs(np.zeros(inf.WIN, np.float32))  # warm-up (graph tracing)
    except Exception as ex:  # service still starts; /health reports the problem
        state["error"] = f"{type(ex).__name__}: {ex}"
    yield


app = FastAPI(title="Intabeh local inference service", lifespan=lifespan, docs_url=None, redoc_url=None)


def fail(code, status):
    return JSONResponse({"ok": False, "error_code": code, "message_ar": ERRORS[code]}, status_code=status)


def preprocess(data, sr):
    """D-014 steps for a live chunk. Returns float32 16 kHz waveform, or None if silent/too short."""
    cfg = state["cfg"]
    y = data.mean(axis=1) if data.ndim == 2 else data
    y = y.astype(np.float32)
    if sr != cfg["target_sample_rate"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg["target_sample_rate"], res_type="soxr_hq").astype(np.float32)
    nz = np.flatnonzero(np.abs(y) >= cfg["edge_trim_threshold"])
    if nz.size == 0:
        return None
    y = y[nz[0]:nz[-1] + 1]
    if len(y) < cfg["min_duration_sec"] * cfg["target_sample_rate"]:
        return None
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    if len(y) < inf.WIN:
        extra = inf.WIN - len(y)
        y = np.pad(y, (extra // 2, extra - extra // 2))
    return y.astype(np.float32)


# LIVE-001 diagnostics: OFF unless INTABEH_DIAG_DIR is set. Saves each received chunk + signal statistics to that
# temporary folder for defect analysis only (never a dataset split, never training/evaluation).
DIAG_DIR = os.environ.get("INTABEH_DIAG_DIR")


def diag_save(body, data, sr, y, ts):
    try:
        d = Path(DIAG_DIR); d.mkdir(parents=True, exist_ok=True)
        stem = ts.replace(":", "").replace("+", "_")
        (d / f"{stem}.wav").write_bytes(body)
        m = data.mean(axis=1)
        stats = {"timestamp": ts, "bytes": len(body), "sample_rate": sr, "channels": int(data.shape[1]),
                 "frames": int(data.shape[0]), "min": float(m.min()), "max": float(m.max()), "mean": float(m.mean()),
                 "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(m.astype(np.float64) ** 2)) + 1e-12)),
                 "clipped_frac": float(np.mean(np.abs(m) >= 0.999)), "finite": bool(np.isfinite(m).all()),
                 "after_preprocess_samples": None if y is None else int(len(y))}
        (d / f"{stem}.json").write_text(json.dumps(stats), encoding="utf-8")
    except Exception:
        pass


def _max_window_dbfs(x, starts):
    return round(max(float(20 * np.log10(np.sqrt(np.mean(np.square(x[s:s + inf.WIN], dtype=np.float64))) + 1e-12))
                     for s in starts), 1)


def live_clip_probs(model, y):
    """Frozen label-free inference with the -60 dBFS gate measured on the YAMNet band (D-025).
    Calls the frozen window_probs unchanged, then keeps only the windows that also pass the frozen gate in the
    >= 125 Hz band (band-limited RMS never exceeds broadband RMS, so this is a subset of the frozen gated windows)."""
    probs, starts, mask = model.window_probs(y)
    band = sosfiltfilt(GATE_SOS, y).astype(np.float32)
    keep = inf.gate_mask(band, starts)[mask]
    level = {"broadband": _max_window_dbfs(y, starts), "band": _max_window_dbfs(band, starts)}
    if not keep.any():
        p = np.zeros(10, np.float32)
        p[model.bg] = 1.0
        return p, 0, level
    return probs[keep].mean(0), int(keep.sum()), level


@app.get("/health")
def health():
    ok = state["model"] is not None
    return JSONResponse({"status": "ok" if ok else "unavailable", "model_loaded": ok, "model_version": MODEL_VERSION,
                         "inference_version": inf.VERSION, "freeze_verified": state["freeze_ok"],
                         "classes": list(LABELS_AR), "labels_ar": LABELS_AR, "error": state["error"]},
                        status_code=200 if ok else 503)


@app.post("/predict")
async def predict(request: Request):
    model = state["model"]
    if model is None:
        return fail("model_unavailable", 503)
    body = await request.body()
    if not body:
        return fail("empty", 400)
    if len(body) > MAX_BYTES:
        return fail("too_large", 413)
    try:
        data, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    except Exception:
        return fail("unreadable", 400)
    if data.size == 0:
        return fail("empty", 400)
    if sr < 16000 or sr > 192000:
        return fail("bad_rate", 400)
    duration = data.shape[0] / sr
    if duration > MAX_SECONDS:
        return fail("too_long", 413)
    if duration < MIN_SECONDS:
        return fail("too_short", 400)
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    y = preprocess(data, sr)
    if DIAG_DIR:
        diag_save(body, data, sr, y, ts)
    base = {"ok": True, "timestamp": ts, "model_version": MODEL_VERSION, "inference_version": inf.VERSION,
            "policy": "frozen Validation-derived thresholds (thresholds_v0.1.json)",
            "input_sample_rate": sr, "duration_sec": round(duration, 3)}
    if y is None:
        return JSONResponse(base | {"predicted_class": "background_other", "label_ar": LABELS_AR["background_other"],
                                    "alert": False, "confidence": None, "silence": True, "gated_windows": 0,
                                    "top_class": None, "probabilities": None})
    try:
        with lock:
            p, k, lv = live_clip_probs(model, y)
            d = model.decide(p, "thresholds")
    except Exception:
        return fail("inference_failed", 500)
    labels = model.labels
    top = int(np.argmax(p))
    pred = labels[d]
    return JSONResponse(base | {
        "input_level_dbfs": lv["broadband"], "band_level_dbfs": lv["band"],
        "predicted_class": pred, "label_ar": LABELS_AR[pred], "alert": pred != "background_other",
        "confidence": round(float(p[d]), 4), "silence": k == 0, "gated_windows": k,
        "top_class": labels[top], "top_label_ar": LABELS_AR[labels[top]], "top_probability": round(float(p[top]), 4),
        "probabilities": {c: round(float(v), 4) for c, v in zip(labels, p)}})


class NoCacheStatic(StaticFiles):
    """Static web files are always revalidated (ETag), so a browser can never keep running an OLD index.html/app.js after an update (no heuristic caching)."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


if PUBLIC.is_dir():
    app.mount("/", NoCacheStatic(directory=str(PUBLIC), html=True), name="web")
