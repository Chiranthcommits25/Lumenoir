"""Safe image decoding helpers (foundation).

Returns Pillow images so no numerical library is required yet. When the
processing pipeline is implemented, this module is the single place to adapt
decoding (colour space, array conversion, resize policy) instead of scattering
conversions across modules.
"""

from __future__ import annotations

import io

from PIL import Image


def decode_image(data: bytes) -> Image.Image:
    """Decode bytes into a fully loaded RGB Pillow image.

    The caller is responsible for having validated ``data`` first; this
    function only guarantees that the returned image is detached from the
    input buffer and has no alpha channel.
    """
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert("RGB")
