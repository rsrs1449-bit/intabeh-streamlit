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
import io
import zipfile
import h5py

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
        pkg = Path(package_dir)
        self.labels = json.loads((pkg / "class_labels.json").read_text(encoding="utf-8"))["labels"]
        th = json.loads((pkg / "thresholds_v0.1.json").read_text(encoding="utf-8"))["thresholds"]
        self.thresholds = [th[c] for c in self.labels]
        self.bg = self.labels.index("background_other")
        # The frozen archive was saved by tf_keras 2.21 with backslash HDF5 group
        # names; Keras 3's generic load_model misses the Normalization variables.
        # Read the SAME frozen weights directly. No training, tuning or resaving.
        self.clf = FrozenClassifier(pkg / "model.keras")
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


class FrozenClassifier:
    """Numpy inference for the exact frozen Normalization -> Dense(ReLU) -> Dense(Softmax)."""
    def __init__(self, model_path):
        with zipfile.ZipFile(model_path) as archive:
            cfg = json.loads(archive.read("config.json"))
            layers = cfg["config"]["layers"]
            sequence = [(layer["class_name"], layer["config"]) for layer in layers]
            if [name for name, _ in sequence] != ["InputLayer", "Normalization", "Dense", "Dropout", "Dense"]:
                raise ValueError("Unexpected frozen classifier architecture")
            if sequence[2][1]["activation"] != "relu" or sequence[4][1]["activation"] != "softmax":
                raise ValueError("Unexpected frozen classifier activations")
            with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as h:
                def get(group, idx):
                    return np.asarray(h[group]["vars"][str(idx)], dtype=np.float32)
                self.mean = get("layers\\normalization", 0)
                self.variance = get("layers\\normalization", 1)
                self.count = get("layers\\normalization", 2)
                self.w1 = get("layers\\dense", 0)
                self.b1 = get("layers\\dense", 1)
                self.w2 = get("layers\\dense_1", 0)
                self.b2 = get("layers\\dense_1", 1)
        if (self.mean.shape != (1024,) or self.variance.shape != (1024,) or
            self.w1.shape != (1024, 256) or self.b1.shape != (256,) or
            self.w2.shape != (256, 10) or self.b2.shape != (10,)):
            raise ValueError("Unexpected frozen classifier weight dimensions")
        if not all(np.all(np.isfinite(a)) for a in (self.mean,self.variance,self.w1,self.b1,self.w2,self.b2)):
            raise ValueError("Non-finite frozen model weights")

    def predict(self, embeddings, verbose=0):
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 1024:
            raise ValueError("Expected YAMNet embeddings [N, 1024]")
        if not len(x):
            return np.zeros((0, 10), dtype=np.float32)
        # Keras Normalization uses sqrt(max(variance, epsilon)) at inference.
        x = (x - self.mean) / np.sqrt(np.maximum(self.variance, 1e-7))
        hidden = np.maximum(x @ self.w1 + self.b1, 0)
        logits = hidden @ self.w2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        exps = np.exp(logits)
        return (exps / exps.sum(axis=1, keepdims=True)).astype(np.float32)
