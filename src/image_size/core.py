"""Read image dimensions from the header without decoding the full image.

Supported formats: PNG, JPEG, GIF, BMP, WebP. Only the minimum number of
bytes required to locate the dimensions are consumed from the input, so this
is suitable for large images where decoding would be expensive or impossible
in memory.

Design decisions
----------------
- ``get_image_size`` accepts either a ``bytes`` buffer or any binary file-like
  object with ``read``. It does NOT accept a filesystem path, because keeping
  the I/O boundary in the caller's hands makes the function trivially testable
  and avoids any question of file lifecycle ownership.
- We read a small fixed-size preamble and dispatch on magic bytes. If the
  preamble is too short for a recognised format, we raise ``ImageSizeError``
  rather than returning a sentinel: callers who asked for dimensions and got
  something with no dimensions deserve a loud failure.
- For JPEG we walk the segment chain rather than assuming the SOF marker is at
  a fixed offset. Real-world JPEGs (especially those produced by cameras with
  EXIF thumbnails first) put SOF arbitrarily far in. This is the awkward edge.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Optional, Tuple, Union


class ImageSizeError(ValueError):
    """Raised when an image's dimensions cannot be determined.

    Subclasses ``ValueError`` so that existing ``except ValueError`` handlers
    still catch it, while giving callers a narrower target if they want to
    distinguish "not an image / unsupported / truncated" from other value
    errors.
    """


@dataclass(frozen=True)
class ImageSizeResult:
    """The dimensions and detected format of a parsed image.

    ``width`` and ``height`` are in pixels. ``format`` is the lower-case
    canonical name of the container (``png``, ``jpeg``, ``gif``, ``bmp``,
    ``webp``) or ``None`` if detection failed before dimensions could be read.
    """

    width: int
    height: int
    format: Optional[str] = None


SUPPORTED_FORMATS: Tuple[str, ...] = ("png", "jpeg", "gif", "bmp", "webp")


def _read_exactly(stream: BinaryIO, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise.

    ``read`` on a real file may return fewer bytes than requested even when
    EOF has not been reached (though in practice it rarely does for regular
    files). Looping until we have ``n`` bytes or a true EOF avoids subtle
    short-read bugs at the cost of a little more code.
    """
    chunks = bytearray()
    while len(chunks) < n:
        chunk = stream.read(n - len(chunks))
        if not chunk:
            raise ImageSizeError(
                f"unexpected end of stream: wanted {n} bytes, got {len(chunks)}"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def _parse_png(buf: BinaryIO) -> ImageSizeResult:
    # PNG: 8-byte signature then IHDR chunk. The IHDR length (4) + type (4)
    # precede the width/height, which are big-endian uint32s.
    _read_exactly(buf, 8)  # consume signature
    # IHDR must be the first chunk.
    length = struct.unpack(">I", _read_exactly(buf, 4))[0]
    chunk_type = _read_exactly(buf, 4)
    if chunk_type != b"IHDR":
        raise ImageSizeError("first PNG chunk is not IHDR")
    if length < 13:
        raise ImageSizeError(f"IHDR chunk too short: {length}")
    data = _read_exactly(buf, 8)  # width, height
    width, height = struct.unpack(">II", data)
    return ImageSizeResult(width=width, height=height, format="png")


def _parse_jpeg(buf: BinaryIO) -> ImageSizeResult:
    # JPEG is a chain of segments. Each starts with 0xFF then a marker byte.
    # Most segments carry a 2-byte big-endian length (which includes the
    # length field itself). We scan for a SOF marker (0xC0..0xCF excluding
    # 0xC4, 0xC8, 0xCC which are DHT, JPGA, DAC) and read the dimensions.
    # SOFn payload: 1 byte precision, 2 bytes height, 2 bytes width, ...
    marker = _read_exactly(buf, 2)
    if marker != b"\xff\xd8":
        raise ImageSizeError("missing JPEG SOI marker")
    while True:
        # Find the next 0xFF. Padding 0xFF bytes between segments are legal.
        ff = _read_exactly(buf, 1)
        while ff == b"\x00":
            ff = _read_exactly(buf, 1)
        if ff != b"\xff":
            raise ImageSizeError("expected JPEG segment marker")
        # Skip fill bytes (0xFF padding).
        m = _read_exactly(buf, 1)
        while m == b"\xff":
            m = _read_exactly(buf, 1)
        code = m[0]
        # Standalone markers without a length payload.
        if code == 0xD9:  # EOI
            raise ImageSizeError("reached EOI without SOF marker")
        if code in (0x01,) or 0xD0 <= code <= 0xD7:
            # RSTn or TEM: no payload.
            continue
        seg_len = struct.unpack(">H", _read_exactly(buf, 2))[0]
        if seg_len < 2:
            raise ImageSizeError(f"JPEG segment length too small: {seg_len}")
        # SOF markers carry dimensions. Exclude non-SOF codes that share the
        # 0xC0..0xCF range.
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            # payload minus the 2-byte length field we already consumed
            payload = _read_exactly(buf, seg_len - 2)
            if len(payload) < 5:
                raise ImageSizeError("SOF segment too short")
            # precision(1) + height(2) + width(2)
            height = struct.unpack(">H", payload[1:3])[0]
            width = struct.unpack(">H", payload[3:5])[0]
            return ImageSizeResult(width=width, height=height, format="jpeg")
        # Skip this segment's payload.
        _read_exactly(buf, seg_len - 2)


def _parse_gif(buf: BinaryIO) -> ImageSizeResult:
    # GIF: 6-byte signature ("GIF87a" or "GIF89a"), then logical screen
    # descriptor: width(2 LE), height(2 LE), packed, bg, aspect.
    sig = _read_exactly(buf, 6)
    if sig[:3] != b"GIF":
        raise ImageSizeError("not a GIF")
    if sig[3:] not in (b"87a", b"89a"):
        raise ImageSizeError(f"unsupported GIF version: {sig[3:]!r}")
    data = _read_exactly(buf, 4)
    width, height = struct.unpack("<HH", data)
    return ImageSizeResult(width=width, height=height, format="gif")


def _parse_bmp(buf: BinaryIO) -> ImageSizeResult:
    # BMP file header: 2 bytes "BM", 4 filesize, 4 reserved, 4 data offset.
    # Then DIB header: 4 bytes header size, 4 bytes width (signed LE),
    # 4 bytes height (signed LE). We only need through the height.
    sig = _read_exactly(buf, 2)
    if sig != b"BM":
        raise ImageSizeError("not a BMP")
    _read_exactly(buf, 12)  # filesize(4) + reserved(4) + offset(4)
    dib_size = struct.unpack("<I", _read_exactly(buf, 4))[0]
    if dib_size < 12:
        raise ImageSizeError(f"BMP DIB header too small: {dib_size}")
    width = struct.unpack("<i", _read_exactly(buf, 4))[0]
    height = struct.unpack("<i", _read_exactly(buf, 4))[0]
    # Negative height means top-down bitmap; the magnitude is the real height.
    return ImageSizeResult(
        width=width,
        height=abs(height),
        format="bmp",
    )


def _parse_webp(buf: BinaryIO) -> ImageSizeResult:
    # RIFF container: "RIFF" + size(4 LE) + "WEBP" + chunk.
    riff = _read_exactly(buf, 4)
    if riff != b"RIFF":
        raise ImageSizeError("not a RIFF container")
    _read_exactly(buf, 4)  # file size, ignore
    webp = _read_exactly(buf, 4)
    if webp != b"WEBP":
        raise ImageSizeError("RIFF form type is not WEBP")
    chunk_fourcc = _read_exactly(buf, 4)
    chunk_size = struct.unpack("<I", _read_exactly(buf, 4))[0]
    if chunk_fourcc == b"VP8 ":
        # Lossy. VP8 frame tag: 3 bytes, then 3 bytes start code (9d 01 2a),
        # then 2 bytes width LE, 2 bytes height LE (each 14-bit).
        if chunk_size < 10:
            raise ImageSizeError("VP8 chunk too short")
        _read_exactly(buf, 3)  # frame tag
        start = _read_exactly(buf, 3)
        if start != b"\x9d\x01\x2a":
            raise ImageSizeError("bad VP8 start code")
        w, h = struct.unpack("<HH", _read_exactly(buf, 4))
        return ImageSizeResult(width=w & 0x3FFF, height=h & 0x3FFF, format="webp")
    if chunk_fourcc == b"VP8L":
        # Lossless. 1 byte signature (0x2f), then 14 bits width-1, 14 bits
        # height-1, packed into a little-endian bitstream across 4 bytes.
        if chunk_size < 5:
            raise ImageSizeError("VP8L chunk too short")
        sig = _read_exactly(buf, 1)
        if sig != b"\x2f":
            raise ImageSizeError("bad VP8L signature")
        bits = _read_exactly(buf, 4)
        w = (bits[0] | ((bits[1] & 0x3F) << 8)) + 1
        h = (((bits[1] >> 6) & 0x03) | (bits[2] << 2) | ((bits[3] & 0x0F) << 10)) + 1
        return ImageSizeResult(width=w, height=h, format="webp")
    if chunk_fourcc == b"VP8X":
        # Extended. 1 byte flags, 3 bytes reserved, 3 bytes canvas width-1 LE,
        # 3 bytes canvas height-1 LE.
        if chunk_size < 10:
            raise ImageSizeError("VP8X chunk too short")
        _read_exactly(buf, 4)  # flags + reserved
        w = int.from_bytes(_read_exactly(buf, 3), "little") + 1
        h = int.from_bytes(_read_exactly(buf, 3), "little") + 1
        return ImageSizeResult(width=w, height=h, format="webp")
    raise ImageSizeError(f"unsupported WebP chunk: {chunk_fourcc!r}")


def _detect_and_parse(buf: BinaryIO) -> ImageSizeResult:
    # Peek the first 12 bytes by reading them; we'll re-feed them via a
    # chained BytesIO so each format parser sees the stream from the start.
    head = _read_exactly(buf, 12)
    import io

    combined = io.BytesIO(head)
    # Chain: combined reads head first, then continues from buf.
    combined = _ChainedStream(head, buf)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return _parse_png(combined)
    if head[:2] == b"\xff\xd8":
        return _parse_jpeg(combined)
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return _parse_gif(combined)
    if head[:2] == b"BM":
        return _parse_bmp(combined)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return _parse_webp(combined)
    raise ImageSizeError("unrecognized image format")


class _ChainedStream:
    """A read-only stream that serves a prefetched prefix then delegates.

    We need to peek the first few bytes to dispatch on the magic number, but
    the format parsers also need to read from byte zero. Re-reading from the
    underlying stream is impossible for pipes/sockets, and rewinding fails on
    non-seekable streams. So we buffer the prefix and hand it out first.
    """

    def __init__(self, prefix: bytes, rest: BinaryIO) -> None:
        self._prefix = memoryview(prefix)
        self._pos = 0
        self._rest = rest

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            tail = self._rest.read()
            out = bytes(self._prefix[self._pos:]) + tail
            self._pos = len(self._prefix)
            return out
        out = bytearray()
        if self._pos < len(self._prefix):
            avail = self._prefix[self._pos:self._pos + n]
            out.extend(avail)
            self._pos += len(avail)
            n -= len(avail)
        if n > 0:
            chunk = self._rest.read(n)
            if chunk:
                out.extend(chunk)
        return bytes(out)


def get_image_size(source: Union[bytes, bytearray, memoryview, BinaryIO]) -> ImageSizeResult:
    """Return the pixel dimensions and format of ``source``.

    ``source`` may be a ``bytes``-like object or any binary stream with a
    ``read`` method. Only the header bytes needed to determine the dimensions
    are consumed; for a file object this means the file pointer is advanced
    past the header but not to EOF.

    Raises ``ImageSizeError`` (a ``ValueError`` subclass) for unrecognized or
    truncated inputs.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        import io

        return _detect_and_parse(io.BytesIO(bytes(source)))
    if hasattr(source, "read"):
        return _detect_and_parse(source)  # type: ignore[arg-type]
    raise TypeError(
        "source must be bytes or a binary file-like object with .read"
    )
