"""Preprocessing reference implementation (Appendix B of the plan)."""
import numpy as np

EPS = 1e-6


def preprocess_world(world):  # world: (21, 3) or (63,)
    P = np.asarray(world, np.float32).reshape(21, 3).copy()
    P -= P[0]                                      # wrist at origin
    s = np.linalg.norm(P[9])                       # wrist -> middle MCP
    P /= max(s, EPS)                               # hand-size invariance
    return P[1:].reshape(-1)                       # (60,)


def preprocess_image(image, wh):  # image: (21,3) normalised, wh=(w,h)
    P = np.asarray(image, np.float32).reshape(21, 3).copy()
    w, h = wh
    P[:, 0] *= w
    P[:, 1] *= h
    P[:, 2] *= w                                    # aspect-ratio correction
    P -= P[0]
    P /= max(np.linalg.norm(P[9]), EPS)
    return P[1:].reshape(-1)


def mirror(P60):  # augmentation: flip x
    X = np.asarray(P60, np.float32).reshape(20, 3).copy()
    X[:, 0] *= -1
    return X.reshape(-1)


def predict_stub(x):
    """Stub predictor for Part 2 until the trained model is wired in."""
    raise NotImplementedError("Model not loaded; wire signo_vN.onnx via onnxruntime.")
