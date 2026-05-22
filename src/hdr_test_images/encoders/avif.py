"""
AVIF encoder via libavif's `avifenc`.

Lossless RGB at depth 12 (AV1's max). Quantize to 12-bit in numpy, write a
16-bit PNG with the 12-bit values left-shifted into the top bits, then let
avifenc rescale back to 12 with `--depth 12`. The validator confirms the
round-trip is bit-exact in 12-bit space.
"""

import subprocess
from pathlib import Path

from ..quantize import TempPng, to_uint, write_png_rgb_depth


def encode(signal_f64, out_path, cicp_primaries, cicp_transfer, bits=12):
    """
    signal_f64: float64 [0,1] transfer-encoded buffer (shape HxWx3).
    cicp_primaries / cicp_transfer: CICP code points (matrix is forced to 0).
    """
    arr_uint = to_uint(signal_f64, bits=bits)
    with TempPng() as tmp_png:
        write_png_rgb_depth(arr_uint, bits=bits, out_path=tmp_png)
        subprocess.run([
            "avifenc",
            "-l",                       # lossless (q=0, YUV 4:4:4 with identity matrix)
            "-y", "444",
            "--range", "full",
            "--depth", str(bits),
            "--cicp", f"{cicp_primaries}/{cicp_transfer}/0",
            str(tmp_png),
            str(out_path),
        ], check=True, capture_output=True)
