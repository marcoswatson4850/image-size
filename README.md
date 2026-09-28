# image_size

Read image pixel dimensions from the header bytes without decoding the image. Supports PNG, JPEG, GIF, BMP, and WebP (lossy, lossless, and extended). Standard library only — no Pillow, no numpy, nothing to install.

## Usage

```python
from image_size import get_image_size, ImageSizeResult

# Build a minimal PNG header in memory
import struct
png = (
    b"\x89PNG\r\n\x1a\n"
    + struct.pack(">I", 13)
    + b"IHDR"
    + struct.pack(">II", 4032, 3024)
    + b"\x08\x06\x00\x00\x00"
    + struct.pack(">I", 0)
)

result: ImageSizeResult = get_image_size(png)

print(result.width, result.height, result.format)
# 4032 3024 png

# Also accepts bytes directly
result = get_image_size(png)
```

## Why this exists

Decoding a 50-megapixel JPEG to learn its dimensions wastes memory and time. The dimensions live in the first few hundred bytes of the file. This library reads only those bytes and stops.

The trade-off: it parses five formats, not every format. It returns an `ImageSizeResult` dataclass (`width`, `height`, `format`) or raises `ImageSizeError`, a `ValueError` subclass. It does not read pixel data, color profiles, or EXIF; if you need those, use a full decoder.

## The awkward edge

JPEG files do not put the dimensions at a fixed offset. A camera JPEG may prepend a large EXIF segment before the `SOF` marker that carries the dimensions. This library walks the segment chain marker by marker until it finds `SOF`, so it handles EXIF-heavy JPEGs correctly — but it will raise on a JPEG whose `SOF` marker it cannot reach within the available bytes (e.g. a truncated download).

## API

- `get_image_size(source) -> ImageSizeResult` — `source` is `bytes`/`bytearray`/`memoryview` or any object with a `.read()` method.
- `ImageSizeResult` — frozen dataclass with `width: int`, `height: int`, `format: str | None`.
- `ImageSizeError` — raised on unrecognized or truncated input; subclasses `ValueError`.
- `SUPPORTED_FORMATS` — tuple of format name strings: `("png", "jpeg", "gif", "bmp", "webp")`.
