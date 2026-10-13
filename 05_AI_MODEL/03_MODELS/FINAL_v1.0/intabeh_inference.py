
"""INTABEH — frozen inference logic v1.0.

Deployment compatibility update:
- Keep the frozen model, labels, thresholds and preprocessing unchanged.
- Try compatible Keras loaders without retraining or changing weights.
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
    """Return window starts and number of full windows."""
    if n < WIN:
        raise ValueError(
            "input shorter than one window; "
            "pad to 15,600 samples first (D-014)"
        )

    starts = list(range(0, n - WIN + 1, HOP))
    full = len(starts)

    if starts[-1] + WIN < n:
        starts.append(n - WIN)

    return starts, full


def gate_mask(y, starts):
    """Return True for windows above the frozen silence gate."""
    rms = np.array([
        np.sqrt(
            np.mean(
                np.square(y[s:s + WIN], dtype=np.float64)
            )
        )
        for s in starts
    ])

    return 20 * np.log10(
        np.maximum(rms, 1e-12)
    ) >= GATE_DBFS


def load_frozen_classifier(model_path):
    """Load the existing classifier without changing its weights."""
    errors = []

    try:
        import tf_keras

        return tf_keras.models.load_model(
            str(model_path),
            compile=False
        )
    except Exception as exc:
        errors.append(
            f"tf_keras: {type(exc).__name__}: {exc}"
        )

    try:
        import keras

        return keras.models.load_model(
            str(model_path),
            compile=False,
            safe_mode=True
        )
    except Exception as exc:
        errors.append(
            f"keras: {type(exc).__name__}: {exc}"
        )

    raise RuntimeError(
        "تعذر تحميل النموذج الأصلي باستخدام واجهتي Keras. "
        "التفاصيل: " + " | ".join(errors)
    )


class IntabehModel:
    def __init__(self, package_dir, tfhub_cache=None):
        root = Path(__file__).resolve().parents[2]

        os.environ.setdefault(
            "TFHUB_CACHE_DIR",
            str(
                tfhub_cache
                or root / ".venv" / "tfhub_cache"
            )
        )
        os.environ.setdefault(
            "TF_CPP_MIN_LOG_LEVEL", "2"
        )

        import tensorflow_hub as hub

        pkg = Path(package_dir)

        self.labels = json.loads(
            (pkg / "class_labels.json").read_text(
                encoding="utf-8"
            )
        )["labels"]

        th = json.loads(
            (pkg / "thresholds_v0.1.json").read_text(
                encoding="utf-8"
            )
        )["thresholds"]

        self.thresholds = [
            th[c] for c in self.labels
        ]

        self.bg = self.labels.index(
            "background_other"
        )

        self.clf = load_frozen_classifier(
            pkg / "model.keras"
        )

        self.yamnet = hub.load(
            YAMNET_HANDLE
        )

    def window_probs(self, y):
        """Return probabilities, window starts and gate mask."""
        y = np.asarray(y, np.float32)

        starts, full = window_plan(len(y))

        _, emb, _ = self.yamnet(y)
        emb = emb.numpy()[:full]

        if len(starts) > full:
            last = starts[-1]

            extra = self.yamnet(
                y[last:last + WIN]
            )[1].numpy()[:1]

            emb = np.vstack([
                emb, extra
            ])

        mask = gate_mask(y, starts)

        if mask.any():
            probs = self.clf.predict(
                emb[mask],
                verbose=0
            )
        else:
            probs = np.zeros(
                (0, 10),
                np.float32
            )

        return probs, starts, mask

    def clip_probs(self, y):
        probs, _, mask = self.window_probs(y)

        if len(probs) == 0:
            p = np.zeros(
                10, np.float32
            )
            p[self.bg] = 1.0
            return p, 0

        return (
            probs.mean(0),
            int(mask.sum())
        )

    def decide(self, p, policy="thresholds"):
        a = int(np.argmax(p))

        if policy == "argmax":
            return a

        t = self.thresholds[a]

        return (
            a
            if t is None or p[a] >= t
            else self.bg
        )
