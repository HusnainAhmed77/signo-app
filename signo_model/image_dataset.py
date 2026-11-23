"""Image preprocessing shared by the public-data notebook and later training code."""

from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
PAD_COLOR = (255, 255, 255)


def prepare_image_for_model(
    image: Image.Image,
    size: tuple[int, int] = IMAGE_SIZE,
    pad_color: tuple[int, int, int] = PAD_COLOR,
) -> Image.Image:
    """Return an EXIF-corrected RGB image resized with aspect ratio and centered padding."""
    if len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
        raise ValueError("size must contain two positive integers.")
    if len(pad_color) != 3 or any(not isinstance(value, int) or not 0 <= value <= 255 for value in pad_color):
        raise ValueError("pad_color must contain three integers in the range 0..255.")

    corrected = ImageOps.exif_transpose(image).convert("RGB")
    resized = ImageOps.contain(corrected, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, pad_color)
    offset = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas
