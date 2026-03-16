#!/usr/bin/env python3
"""
Generate HDR HEIF test images for validating the BT.2100 OOTF fix in GTK4.

Produces files encoding the same intended display appearance:
  test_reference.jpg    — sRGB JPEG ground truth
  test_srgb.heif        — HEIF with CICP 1/13/0/1 (sRGB)
  test_hlg.heif         — HEIF with CICP 9/18/0/1 (HLG BT.2020, 1000-nit γ=1.2)
  test_hlg_p3.heif      — HEIF with CICP 12/18/0/1 (HLG Display P3, 1000-nit γ=1.2)
  test_pq.heif          — HEIF with CICP 9/16/0/1 (PQ BT.2020, SDR white=203 nits)

With correct OOTF / PQ implementations, all files should look identical.

Dependencies: numpy, Pillow
External tools: heif-enc (libheif), ffmpeg
"""

import numpy as np
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Transfer functions — matching gtk/gdk/gdkcolordefs.h exactly
# ---------------------------------------------------------------------------

def srgb_oetf(v):
    """gdkcolordefs.h:33-38"""
    v = np.asarray(v, dtype=np.float64)
    return np.where(
        np.abs(v) > 0.0031308,
        np.sign(v) * (1.055 * np.abs(v) ** (1.0 / 2.4) - 0.055),
        12.92 * v,
    )

def hlg_oetf(v):
    """gdkcolordefs.h:140-151"""
    v = np.asarray(v, dtype=np.float64)
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            np.abs(v) <= 1.0 / 12.0,
            np.sign(v) * np.sqrt(3.0 * np.abs(v)),
            np.sign(v) * (a * np.log(np.maximum(12.0 * np.abs(v) - b, 1e-30)) + c),
        )

def pq_oetf(v):
    """SMPTE ST 2084 OETF: linear nits → PQ signal."""
    v = np.asarray(v, dtype=np.float64)
    Lm = np.maximum(v / 10000.0, 0.0)
    m1 = 0.1593017578125    # 2610 / 16384
    m2 = 78.84375           # 2523 / 32 * 128
    c1 = 0.8359375          # 3424 / 4096
    c2 = 18.8515625         # 2413 / 128
    c3 = 18.6875            # 2392 / 128
    Lm_m1 = Lm ** m1
    return ((c1 + c2 * Lm_m1) / (1.0 + c3 * Lm_m1)) ** m2

# ---------------------------------------------------------------------------
# Color space matrices — matching gdkcolordefs.h:231-235, 225-229
# ---------------------------------------------------------------------------

SRGB_TO_REC2020 = np.array([
    [0.627504, 0.329275, 0.043303],
    [0.069108, 0.919519, 0.011360],
    [0.016394, 0.088011, 0.895380],
], dtype=np.float64)

SRGB_TO_P3 = np.array([
    [0.822462, 0.177538, 0.000000],
    [0.033194, 0.966806, 0.000000],
    [0.017083, 0.072397, 0.910520],
], dtype=np.float64)

# BT.2020 luma coefficients (BT.2100 Table 5)
REC2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)

# Display P3 luma coefficients (Y row of P3-D65 RGB-to-XYZ matrix)
P3_LUMA = np.array([0.2290, 0.6917, 0.0793], dtype=np.float64)

# ---------------------------------------------------------------------------
# HLG OOTF / inverse OOTF — BT.2100 Section 5.3
# ---------------------------------------------------------------------------

def hlg_inverse_ootf(display_linear, luma_coeffs, system_gamma=0.78, luminance_scale=1.0):
    """
    Inverse OOTF: display-referred linear → scene-referred linear.
    Matches GTK's apply_cicp_inverse_ootf (gdkcolorstate.c:783-809):
      Yd = dot(luma_coeffs, display)
      scale = (1/luminance_scale)^(1/gamma) * Yd^(1/gamma - 1)
      scene = display * scale
    luminance_scale=1.0 for sRGB target, 1000/203 for rec2100 target.
    """
    display = np.asarray(display_linear, dtype=np.float64)
    Yd = np.sum(display * luma_coeffs, axis=-1, keepdims=True)
    Yd = np.maximum(Yd, 0.0)
    inv_ls = (1.0 / luminance_scale) ** (1.0 / system_gamma)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(Yd > 0.0, inv_ls * Yd ** (1.0 / system_gamma - 1.0), 0.0)
    return display * scale

# ---------------------------------------------------------------------------
# Test chart generation
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 320

# Patch definitions: (name, linear_srgb_rgb)
PATCHES_ROW1 = [
    ("White",     (1.000, 1.000, 1.000)),
    ("18% Gray",  (0.180, 0.180, 0.180)),
    ("50% Gray",  (0.500, 0.500, 0.500)),
    ("Red",       (0.800, 0.100, 0.100)),
    ("Green",     (0.100, 0.600, 0.100)),
    ("Blue",      (0.100, 0.100, 0.800)),
    ("Cyan",      (0.100, 0.600, 0.600)),
    ("Magenta",   (0.600, 0.100, 0.600)),
]

PATCHES_ROW2 = [
    ("Yellow",    (0.600, 0.600, 0.100)),
    ("Skin",      (0.350, 0.200, 0.140)),
    ("5% Gray",   (0.050, 0.050, 0.050)),
    ("10% Gray",  (0.100, 0.100, 0.100)),
    ("25% Gray",  (0.250, 0.250, 0.250)),
    ("75% Gray",  (0.750, 0.750, 0.750)),
    ("90% Gray",  (0.900, 0.900, 0.900)),
    ("Black",     (0.000, 0.000, 0.000)),
]


def make_chart():
    """Create 640x320 test chart as float64 linear sRGB [0,1]."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    # Top band: 16-step grayscale ramp (y=0..159)
    for i in range(16):
        x0 = i * (WIDTH // 16)
        x1 = (i + 1) * (WIDTH // 16)
        level = i / 15.0  # linear sRGB 0.0 to 1.0
        img[0:160, x0:x1, :] = level

    # Middle band: color patches (y=160..319)
    patch_w = WIDTH // 8   # 80px
    patch_h = 80

    for col, (name, rgb) in enumerate(PATCHES_ROW1):
        x0 = col * patch_w
        x1 = (col + 1) * patch_w
        img[160:160 + patch_h, x0:x1, :] = rgb

    for col, (name, rgb) in enumerate(PATCHES_ROW2):
        x0 = col * patch_w
        x1 = (col + 1) * patch_w
        img[160 + patch_h:160 + 2 * patch_h, x0:x1, :] = rgb

    return img


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_jpeg(chart_linear, output_path):
    """Apply sRGB OETF, quantize to uint8, save JPEG."""
    srgb = srgb_oetf(chart_linear)
    srgb = np.clip(srgb, 0.0, 1.0)
    img_u8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img_u8, "RGB").save(output_path, quality=100, subsampling=0)
    print(f"  Written: {output_path}")


def encode_srgb_heif(chart_linear, output_path):
    """sRGB OETF → uint8 PNG → heif-enc with CICP 1/13/0/1."""
    srgb = srgb_oetf(chart_linear)
    srgb = np.clip(srgb, 0.0, 1.0)
    img_u8 = (srgb * 255.0 + 0.5).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_png = f.name
    try:
        Image.fromarray(img_u8, "RGB").save(tmp_png)
        subprocess.run([
            "heif-enc",
            "--colour_primaries", "1",
            "--transfer_characteristic", "13",
            "--matrix_coefficients", "0",
            "--full_range_flag", "1",
            "-L",
            "-o", str(output_path),
            tmp_png,
        ], check=True, capture_output=True)
        print(f"  Written: {output_path}")
    finally:
        Path(tmp_png).unlink(missing_ok=True)


def encode_hlg_heif(chart_linear, output_path, system_gamma, luminance_scale,
                    gamut_matrix=None, luma_coeffs=None, cicp_primaries="9"):
    """
    Linear sRGB → target gamut → inverse OOTF → HLG OETF → 16-bit → ffmpeg PNG → heif-enc.
    Default CICP 9/18/0/1 (BT.2020 primaries, HLG transfer, RGB matrix, full range).
    """
    if gamut_matrix is None:
        gamut_matrix = SRGB_TO_REC2020
    if luma_coeffs is None:
        luma_coeffs = REC2020_LUMA

    # 1. Gamut convert: linear sRGB → target linear
    flat = chart_linear.reshape(-1, 3)
    target_linear = (gamut_matrix @ flat.T).T
    target_linear = np.maximum(target_linear, 0.0)
    target_display = target_linear.reshape(HEIGHT, WIDTH, 3)

    # 2. Inverse OOTF: display-referred → scene-referred
    target_scene = hlg_inverse_ootf(target_display,
                                    luma_coeffs=luma_coeffs,
                                    system_gamma=system_gamma,
                                    luminance_scale=luminance_scale)

    # 3. HLG OETF: scene-linear → HLG signal
    hlg_signal = hlg_oetf(target_scene)
    hlg_signal = np.clip(hlg_signal, 0.0, 1.0)

    # 4. Quantize to 16-bit and pipe through ffmpeg for 16-bit PNG
    img_u16 = (hlg_signal * 65535.0 + 0.5).astype(np.uint16)
    raw_data = img_u16.astype(">u2").tobytes()  # rgb48be

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_png = f.name
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48be",
            "-s", f"{WIDTH}x{HEIGHT}",
            "-i", "pipe:",
            "-update", "1",
            tmp_png,
        ], input=raw_data, check=True, capture_output=True)

        subprocess.run([
            "heif-enc",
            "-b", "10",
            "--colour_primaries", cicp_primaries,
            "--transfer_characteristic", "18",
            "--matrix_coefficients", "0",
            "--full_range_flag", "1",
            "-L",
            "-o", str(output_path),
            tmp_png,
        ], check=True, capture_output=True)
        print(f"  Written: {output_path} (γ={system_gamma:.3f}, LS={luminance_scale:.3f})")
    finally:
        Path(tmp_png).unlink(missing_ok=True)


def encode_pq_heif(chart_linear, output_path, sdr_white_nits=203.0):
    """
    Linear sRGB → BT.2020 → scale to nits → PQ OETF → 16-bit → ffmpeg PNG → heif-enc.
    CICP 9/16/0/1 (BT.2020 primaries, PQ transfer, RGB matrix, full range).
    SDR white is mapped to sdr_white_nits (default 203 per BT.2408).
    """

    # 1. Gamut convert: linear sRGB → linear BT.2020
    flat = chart_linear.reshape(-1, 3)
    rec2020_linear = (SRGB_TO_REC2020 @ flat.T).T
    rec2020_linear = np.maximum(rec2020_linear, 0.0)
    rec2020_nits = rec2020_linear.reshape(HEIGHT, WIDTH, 3) * sdr_white_nits

    # 2. PQ OETF: linear nits → PQ signal
    pq_signal = pq_oetf(rec2020_nits)
    pq_signal = np.clip(pq_signal, 0.0, 1.0)

    # 3. Quantize to 16-bit and pipe through ffmpeg for 16-bit PNG
    img_u16 = (pq_signal * 65535.0 + 0.5).astype(np.uint16)
    raw_data = img_u16.astype(">u2").tobytes()  # rgb48be

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_png = f.name
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48be",
            "-s", f"{WIDTH}x{HEIGHT}",
            "-i", "pipe:",
            "-update", "1",
            tmp_png,
        ], input=raw_data, check=True, capture_output=True)

        subprocess.run([
            "heif-enc",
            "-b", "10",
            "--colour_primaries", "9",
            "--transfer_characteristic", "16",
            "--matrix_coefficients", "0",
            "--full_range_flag", "1",
            "-L",
            "-o", str(output_path),
            tmp_png,
        ], check=True, capture_output=True)
        print(f"  Written: {output_path} (PQ, SDR white={sdr_white_nits:.0f} nits)")
    finally:
        Path(tmp_png).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent
    print("Generating HLG OOTF test images...")
    print(f"  Output directory: {out_dir}")
    print()

    # Check for heif-enc
    if not shutil.which("heif-enc"):
        print("ERROR: heif-enc not found. Install libheif to generate HEIF files.")
        sys.exit(1)

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Install ffmpeg for 16-bit PNG encoding.")
        sys.exit(1)

    chart = make_chart()

    print("Encoding files:")
    encode_jpeg(chart, out_dir / "test_reference.jpg")
    encode_srgb_heif(chart, out_dir / "test_srgb.heif")
    encode_hlg_heif(chart, out_dir / "test_hlg.heif",
                    system_gamma=1.2, luminance_scale=1000.0 / 203.0)
    encode_hlg_heif(chart, out_dir / "test_hlg_p3.heif",
                    system_gamma=1.2, luminance_scale=1000.0 / 203.0,
                    gamut_matrix=SRGB_TO_P3, luma_coeffs=P3_LUMA,
                    cicp_primaries="12")
    encode_pq_heif(chart, out_dir / "test_pq.heif")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
