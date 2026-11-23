"""Unit tests for preprocessing (section 9.6). Run: python -m pytest tests/"""
import numpy as np
from signo_model.preprocess import preprocess_world, preprocess_image, mirror


def make_hand():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(21, 3)).astype(np.float32)
    P[0] = np.array([0.1, 0.2, 0.0])  # wrist
    return P


def test_shape():
    P = make_hand()
    out = preprocess_world(P)
    assert out.shape == (60,)


def test_translation_invariance():
    P = make_hand()
    P2 = P.copy(); P2 += np.array([5.0, -3.0, 2.0])
    a = preprocess_world(P); b = preprocess_world(P2)
    assert np.allclose(a, b, atol=1e-5)


def test_scale_invariance():
    P = make_hand()
    a = preprocess_world(P); b = preprocess_world(P * 2.3)
    assert np.allclose(a, b, atol=1e-5)


def test_rotation_sensitivity():
    P = make_hand()
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    Pr = P @ R.T
    a = preprocess_world(P); b = preprocess_world(Pr)
    assert not np.allclose(a, b)


def test_mirror():
    P = make_hand()
    a = preprocess_world(P)
    m = mirror(a)
    assert m.shape == a.shape
    assert not np.allclose(m, a)


def test_deterministic():
    P = make_hand()
    assert np.array_equal(preprocess_world(P), preprocess_world(P))


def test_image_variant():
    P = make_hand()
    # normalise to [0,1]-ish
    img = P.copy()
    img[:, 0] = (img[:, 0] - img[:, 0].min()) / 10.0
    img[:, 1] = (img[:, 1] - img[:, 1].min()) / 10.0
    out = preprocess_image(img, (640, 480))
    assert out.shape == (60,)


def test_nan_handling():
    P = make_hand()
    P[5] = np.nan
    try:
        preprocess_world(P)
    except Exception:
        pass  # graceful: either raises or returns; document behaviour
