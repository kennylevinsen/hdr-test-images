"""HEIF encoder via libheif's `heif-enc`. Bumped to 12-bit for HDR variants."""

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ..quantize import TempPng, to_uint, write_png_rgb_depth


def encode_srgb(srgb_signal_f64, out_path):
    """8-bit sRGB HEIF — CICP 1/13/0/1."""
    arr_u8 = to_uint(srgb_signal_f64, bits=8)
    with TempPng() as tmp_png:
        Image.fromarray(arr_u8, "RGB").save(tmp_png)
        subprocess.run([
            "heif-enc",
            "--colour_primaries", "1",
            "--transfer_characteristic", "13",
            "--matrix_coefficients", "0",
            "--full_range_flag", "1",
            "-L",
            "-o", str(out_path),
            str(tmp_png),
        ], check=True, capture_output=True)


def encode_hdr(signal_f64, out_path, cicp_primaries, cicp_transfer, bits=12):
    """
    HDR HEIF (HLG or PQ, any primaries). CICP M=0 (RGB), R=1 (full).
    Quantizes to `bits` and writes a 16-bit RGB PNG intermediate
    (heif-enc reads PNG and downconverts to -b bits via libheif).
    """
    arr_uint = to_uint(signal_f64, bits=bits)
    with TempPng() as tmp_png:
        write_png_rgb_depth(arr_uint, bits=bits, out_path=tmp_png)
        subprocess.run([
            "heif-enc",
            "-b", str(bits),
            "--colour_primaries", str(cicp_primaries),
            "--transfer_characteristic", str(cicp_transfer),
            "--matrix_coefficients", "0",
            "--full_range_flag", "1",
            "-L",
            "-o", str(out_path),
            str(tmp_png),
        ], check=True, capture_output=True)
