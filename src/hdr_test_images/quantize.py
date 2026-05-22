"""
Quantization and intermediate-PNG writers shared by encoders.

All encoders consume the same transfer-encoded float [0,1] buffer and only
differ in how they quantize / serialize from here on. Centralizing the
quantization step lets the validator predict the exact integer pixel values
each encoder receives — so a mismatch with the decoded file isolates the bug
to the encoder, not to our reference.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def to_uint(arr_float01, bits):
    """Float [0,1] → unsigned int [0, 2^bits - 1] with round-half-up."""
    if bits not in (8, 10, 12, 16):
        raise ValueError(f"unsupported bit depth: {bits}")
    max_val = (1 << bits) - 1
    out = np.clip(arr_float01, 0.0, 1.0) * max_val + 0.5
    dtype = np.uint8 if bits == 8 else np.uint16
    return out.astype(dtype)


def write_png_8bit(arr_u8, out_path):
    """Write 8-bit RGB PNG via Pillow."""
    Image.fromarray(arr_u8, "RGB").save(out_path)


def write_png_rgb48be(arr_u16, out_path):
    """
    Write 16-bit RGB PNG via ffmpeg from raw rgb48be bytes.
    Returns the output path. Uses ffmpeg because Pillow's 16-bit RGB support is
    quirky across versions, while ffmpeg's rgb48be → PNG path is dependable.
    """
    h, w, c = arr_u16.shape
    assert c == 3 and arr_u16.dtype == np.uint16
    raw_data = arr_u16.astype(">u2").tobytes()

    subprocess.run([
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48be",
        "-s", f"{w}x{h}",
        "-i", "pipe:",
        "-update", "1",
        str(out_path),
    ], input=raw_data, check=True, capture_output=True)


def expand_to_u16(arr_uint, bits):
    """
    Expand integer samples at `bits` precision into uint16 via bit-replication,
    so that the max input value (2^bits - 1) maps to 65535. This is the
    canonical "expand low-bit-depth to high-bit-depth without value drift"
    formula used by PNG/avifdec/etc.
        out = (in << (16 - bits)) | (in >> (2*bits - 16))
    Simple left-shift would lose precision at the high end — avifenc's
    16→12 round-trip then drops the max value by 1 LSB.
    """
    if bits == 16:
        return arr_uint.astype(np.uint16)
    if bits < 8 or bits > 16:
        raise ValueError(f"bit depth out of range: {bits}")
    arr_u32 = arr_uint.astype(np.uint32)
    high = arr_u32 << (16 - bits)
    low = arr_u32 >> (2 * bits - 16) if bits >= 8 else np.uint32(0)
    return (high | low).astype(np.uint16)


def write_png_rgb_depth(arr_uint, bits, out_path):
    """Write PNG at the given bit depth. 8-bit uses Pillow; 10/12/16 uses ffmpeg.

    For 10/12-bit, values are bit-replicated into uint16 so that downstream
    encoders' 16→target rescaling is bit-exact for every value.
    """
    if bits == 8:
        write_png_8bit(arr_uint, out_path)
        return
    if bits not in (10, 12, 16):
        raise ValueError(f"unsupported bit depth: {bits}")
    arr_u16 = expand_to_u16(arr_uint, bits)
    write_png_rgb48be(arr_u16, out_path)


class TempPng:
    """Context manager: returns a tempfile path for a .png; deletes on exit."""

    def __enter__(self):
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        self.path = Path(f.name)
        return self.path

    def __exit__(self, *exc):
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass
