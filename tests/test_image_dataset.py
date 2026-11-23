import numpy as np
import pytest
from PIL import Image

from signo_model.image_dataset import IMAGE_SIZE, PAD_COLOR, prepare_image_for_model


def test_prepare_image_returns_fixed_size_rgb_with_centered_padding():
    source = Image.new("RGB", (40, 20), (10, 20, 30))

    prepared = prepare_image_for_model(source)

    assert prepared.size == IMAGE_SIZE
    assert prepared.mode == "RGB"
    pixels = np.asarray(prepared)
    assert tuple(pixels[0, 0]) == PAD_COLOR
    assert tuple(pixels[112, 112]) != PAD_COLOR
    assert np.all(pixels[:56] == np.array(PAD_COLOR, dtype=np.uint8))
    assert np.all(pixels[168:] == np.array(PAD_COLOR, dtype=np.uint8))


def test_prepare_image_supports_custom_size_and_color():
    prepared = prepare_image_for_model(Image.new("L", (20, 40), 127), (80, 80), (1, 2, 3))

    assert prepared.size == (80, 80)
    assert prepared.mode == "RGB"
    assert tuple(np.asarray(prepared)[0, 0]) == (1, 2, 3)


@pytest.mark.parametrize(
    ("size", "pad_color"),
    [
        ((0, 224), PAD_COLOR),
        ((224,), PAD_COLOR),
        (IMAGE_SIZE, (256, 0, 0)),
        (IMAGE_SIZE, (255, 255)),
    ],
)
def test_prepare_image_rejects_invalid_configuration(size, pad_color):
    with pytest.raises(ValueError):
        prepare_image_for_model(Image.new("RGB", (10, 10)), size, pad_color)
