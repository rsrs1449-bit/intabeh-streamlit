"""INTABEH — frozen inference logic v1.0 (Day 4; D-023). Shared by evaluation, sanity checks and the Day 5 app.

LABEL-FREE (deployment-realistic): the true class is never used at inference time.
  waveform (16 kHz mono float32, D-014 preprocessing) -> 15,600-sample windows, hop 7,680 + end-aligned window
  -> silence gate: windows with RMS < -60 dBFS are not classified (D-014 app gate)
  -> frozen YAMNet (TF-Hub google/yamnet/1) 1024-d embedding per window
  -> classifier (BEST_MODEL / FINAL package) -> 10 probabilities per window
  -> clip probability = mean over gated windows (no gated window -> no alert / background_other)
  -> decision: argmax, or the frozen Validation-derived threshold policy:
     alert class c if argmax == c and p_c >= t_c (t_c None -> argmax default); otherwise background_other.
"""

import json
import os
from pathlib import Path

import numpy as np

WIN, HOP, SR = 15600, 7680, 16000
GATE_DBFS = -60.0
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
VERSION = "1.0"


def window_plan(n):
    """Window starts (full windows at k*HOP plus one end-aligned window) and the number of full windows."""
    if n < WIN:
        raise ValueError("input shorter than one window; pad to 15,600 samples first (D-014)")
    starts = list(range(0, n - WIN + 1, HOP))
    full = len(starts)
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return starts, full


def gate_mask(y, starts):
    rms = np.array([np.sqrt(np.mean(np.square(y[s:s + WIN], dtype=np.float64))) for s in starts])
    return 20 * np.log10(np.maximum(rms, 1e-12)) >= GATE_DBFS


class IntabehModel:
    def __init__(self, package_dir, tfhub_cache=None):
        root = Path(__file__).resolve().parents[2]
        os.environ.setdefault("TFHUB_CACHE_DIR", str(tfhub_cache or root / ".venv" / "tfhub_cache"))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow_hub as hub
        import tf_keras as keras
        pkg = Path(package_dir)
        self.labels = json.loads((pkg / "class_labels.json").read_text(encoding="utf-8"))["labels"]
        th = json.loads((pkg / "thresholds_v0.1.json").read_text(encoding="utf-8"))["thresholds"]
        self.thresholds = [th[c] for c in self.labels]
        self.bg = self.labels.index("background_other")
        self.clf = keras.models.load_model(
            pkg / "model.keras",
            compile=False
        )
        self.yamnet = hub.load(YAMNET_HANDLE)

    def window_probs(self, y):
        """Return (probabilities for gated windows [k x 10], window starts, gate mask)."""
        y = np.asarray(y, np.float32)
        starts, full = window_plan(len(y))
        _, emb, _ = self.yamnet(y)
        emb = emb.numpy()[:full]
        if len(starts) > full:
            emb = np.vstack([emb, self.yamnet(y[starts[-1]:starts[-1] + WIN])[1].numpy()[:1]])
        mask = gate_mask(y, starts)
        probs = self.clf.predict(emb[mask], verbose=0) if mask.any() else np.zeros((0, 10), np.float32)
        return probs, starts, mask

    def clip_probs(self, y):
        probs, _, mask = self.window_probs(y)
        if len(probs) == 0:
            p = np.zeros(10, np.float32)
            p[self.bg] = 1.0
            return p, 0
        return probs.mean(0), int(mask.sum())

    def decide(self, p, policy="thresholds"):
        a = int(np.argmax(p))
        if policy == "argmax":
            return a
        t = self.thresholds[a]
        return a if t is None or p[a] >= t else self.bg
