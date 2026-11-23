"""Prediction helpers (stub until model is trained)."""
import numpy as np


def predict(logits):
    """Return (label_index, confidence) from raw logits array."""
    x = np.asarray(logits, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    # softmax
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    idx = int(np.argmax(p, axis=1)[0])
    conf = float(p[0, idx])
    return idx, conf
