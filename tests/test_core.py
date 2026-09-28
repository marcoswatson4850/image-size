import io
import struct
import unittest

from image_size import get_image_size, ImageSizeError, ImageSizeResult, SUPPORTED_FORMATS


def png_bytes(width, height):
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_data = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    crc = struct.pack(">I", 0)
    return sig + ihdr_len + ihdr_type + ihdr_data + crc


def gif_bytes(width, height, version=b"89a"):
    return b"GIF" + version + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def bmp_bytes(width, height):
    header = b"BM" + struct.pack("<I", 26) + b"\x00\x00\x00\x00" + struct.pack("<I", 14)
    dib = struct.pack("<IiiHHIIiiII", 12, width, height, 1, 24, 0, 0, 0, 0, 0, 0)
    return header + dib


def jpeg_sof_bytes(width, height, marker=0xC0):
    soi = b"\xff\xd8"
    # One APP0 segment before SOF to exercise segment skipping.
    app0_len = 16
    app0 = b"\xff\xe0" + struct.pack(">H", app0_len) + b"JFIF\x00" + b"\x01\x01\x00\x00\x01"
    # Pad APP0 to exactly app0_len bytes (including length field).
    app0 += b"\x00" * (app0_len - 2 - (len(app0) - 4))
    # SOF segment.
    sof_payload = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    sof_len = 2 + len(sof_payload)
    sof = bytes([0xFF, marker]) + struct.pack(">H", sof_len) + sof_payload
    return soi + app0 + sof


def webp_lossy_bytes(width, height):
    w = width & 0x3FFF
    h = height & 0x3FFF
    vp8_payload = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", w, h)
    chunk = b"VP8 " + struct.pack("<I", len(vp8_payload)) + vp8_payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def webp_lossless_bytes(width, height):
    w = width - 1
    h = height - 1
    b0 = w & 0xFF
    b1 = ((w >> 8) & 0x3F) | ((h & 0x03) << 6)
    b2 = (h >> 2) & 0xFF
    b3 = (h >> 10) & 0x0F
    payload = b"\x2f" + bytes([b0, b1, b2, b3])
    chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def webp_extended_bytes(width, height):
    w = width - 1
    h = height - 1
    payload = b"\x00\x00\x00\x00" + w.to_bytes(3, "little") + h.to_bytes(3, "little")
    chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


class TestPNG(unittest.TestCase):
    def test_basic(self):
        r = get_image_size(png_bytes(640, 480))
        self.assertEqual((r.width, r.height), (640, 480))
        self.assertEqual(r.format, "png")

    def test_large(self):
        r = get_image_size(png_bytes(2_000_000_000, 1))
        self.assertEqual(r.width, 2_000_000_000)


class TestJPEG(unittest.TestCase):
    def test_basic(self):
        r = get_image_size(jpeg_sof_bytes(800, 600))
        self.assertEqual((r.width, r.height), (800, 600))
        self.assertEqual(r.format, "jpeg")

    def test_sof2_marker(self):
        r = get_image_size(jpeg_sof_bytes(100, 200, marker=0xCF))
        self.assertEqual((r.width, r.height), (100, 200))

    def test_ff_fill_bytes_between_segments(self):
        data = b"\xff\xd8" + b"\xff\xff\xff\xe0" + struct.pack(">H", 6) + b"JF" + b"\x00\x00"
        data += jpeg_sof_bytes(50, 50)[2:]
        r = get_image_size(data)
        self.assertEqual((r.width, r.height), (50, 50))


class TestGIF(unittest.TestCase):
    def test_89a(self):
        r = get_image_size(gif_bytes(100, 200))
        self.assertEqual((r.width, r.height), (100, 200))
        self.assertEqual(r.format, "gif")

    def test_87a(self):
        r = get_image_size(gif_bytes(1, 1, version=b"87a"))
        self.assertEqual((r.width, r.height), (1, 1))


class TestBMP(unittest.TestCase):
    def test_basic(self):
        r = get_image_size(bmp_bytes(320, 240))
        self.assertEqual((r.width, r.height), (320, 240))
        self.assertEqual(r.format, "bmp")

    def test_top_down_negative_height(self):
        header = b"BM" + struct.pack("<I", 26) + b"\x00\x00\x00\x00" + struct.pack("<I", 14)
        dib = struct.pack("<IiiHHIIiiII", 12, 100, -100, 1, 24, 0, 0, 0, 0, 0, 0)
        r = get_image_size(header + dib)
        self.assertEqual((r.width, r.height), (100, 100))


class TestWebP(unittest.TestCase):
    def test_lossy(self):
        r = get_image_size(webp_lossy_bytes(1920, 1080))
        self.assertEqual((r.width, r.height), (1920, 1080))
        self.assertEqual(r.format, "webp")

    def test_lossless(self):
        r = get_image_size(webp_lossless_bytes(256, 256))
        self.assertEqual((r.width, r.height), (256, 256))

    def test_extended(self):
        r = get_image_size(webp_extended_bytes(4096, 2048))
        self.assertEqual((r.width, r.height), (4096, 2048))


class TestStreamInput(unittest.TestCase):
    def test_file_like(self):
        buf = io.BytesIO(png_bytes(99, 77))
        r = get_image_size(buf)
        self.assertEqual((r.width, r.height), (99, 77))

    def test_memoryview(self):
        r = get_image_size(memoryview(png_bytes(10, 20)))
        self.assertEqual((r.width, r.height), (10, 20))


class TestErrors(unittest.TestCase):
    def test_unknown_format(self):
        with self.assertRaises(ImageSizeError):
            get_image_size(b"hello world!!!")

    def test_truncated(self):
        with self.assertRaises(ImageSizeError):
            get_image_size(b"\x89PNG\r\n\x1a")

    def test_is_value_error(self):
        self.assertTrue(issubclass(ImageSizeError, ValueError))

    def test_bad_type(self):
        with self.assertRaises(TypeError):
            get_image_size(123)


class TestExports(unittest.TestCase):
    def test_result_fields(self):
        r = ImageSizeResult(width=1, height=2, format="png")
        self.assertEqual(r.width, 1)
        self.assertEqual(r.height, 2)
        self.assertEqual(r.format, "png")

    def test_supported_formats(self):
        self.assertIn("png", SUPPORTED_FORMATS)
        self.assertIn("jpeg", SUPPORTED_FORMATS)


if __name__ == "__main__":
    unittest.main()
