"""
Transfer functions and gamut matrices for HDR encoding.

All math matches GTK4's gdk/gdkcolordefs.h line-for-line. Validated by the property
that, in the existing test set, JPEG/HEIF-sRGB/HEIF-HLG/HEIF-HLG-P3/HEIF-PQ all
look identical on a 1000-nit display.
"""

import numpy as np


# ---------------------------------------------------------------------------
# OETFs (encoders: linear → non-linear signal)
# ---------------------------------------------------------------------------

def srgb_oetf(v):
    """sRGB OETF (gdkcolordefs.h:33-38)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(
        np.abs(v) > 0.0031308,
        np.sign(v) * (1.055 * np.abs(v) ** (1.0 / 2.4) - 0.055),
        12.92 * v,
    )


def hlg_oetf(v):
    """HLG OETF (gdkcolordefs.h:140-151). Scene-linear in [0,1] → HLG signal [0,1]."""
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
    """SMPTE ST 2084 OETF: linear nits → PQ signal [0,1]."""
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
# Gamut conversion matrices (linear RGB → linear RGB, D65)
# Match gdk/gdkcolordefs.h:231-235, 225-229
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
# HLG inverse OOTF (BT.2100 Section 5.3)
# ---------------------------------------------------------------------------

def hlg_inverse_ootf(display_linear, luma_coeffs, system_gamma=0.78, luminance_scale=1.0):
    """
    display-referred linear → scene-referred linear.
    Matches GTK's apply_cicp_inverse_ootf (gdkcolorstate.c:783-809).
    luminance_scale = 1000/203 for BT.2100 1000-nit target paired with SDR-white=203.
    """
    display = np.asarray(display_linear, dtype=np.float64)
    Yd = np.sum(display * luma_coeffs, axis=-1, keepdims=True)
    Yd = np.maximum(Yd, 0.0)
    inv_ls = (1.0 / luminance_scale) ** (1.0 / system_gamma)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(Yd > 0.0, inv_ls * Yd ** (1.0 / system_gamma - 1.0), 0.0)
    return display * scale


# ---------------------------------------------------------------------------
# High-level pipelines: linear sRGB chart → transfer-encoded float [0,1]
# These produce the exact float buffer that gets quantized and serialized.
# Shared between every encoder so they cannot disagree on the math.
# ---------------------------------------------------------------------------

def linear_srgb_to_hlg_signal(chart_linear_srgb, gamut_matrix, luma_coeffs,
                              system_gamma=1.2, luminance_scale=1000.0 / 203.0):
    """
    Linear sRGB → target gamut → HLG inverse OOTF → HLG OETF.
    Returns float64 in [0,1] (HLG signal), shape same as input.

    system_gamma matches the gtk source convention: it's used in the exponent
    1/system_gamma - 1 inside hlg_inverse_ootf, so passing 1.2 reproduces the
    inverse of the BT.2100 1.2-gamma OOTF.
    """
    flat = chart_linear_srgb.reshape(-1, 3)
    target_linear = (gamut_matrix @ flat.T).T
    target_linear = np.maximum(target_linear, 0.0)
    target_display = target_linear.reshape(chart_linear_srgb.shape)

    target_scene = hlg_inverse_ootf(
        target_display,
        luma_coeffs=luma_coeffs,
        system_gamma=system_gamma,
        luminance_scale=luminance_scale,
    )

    hlg_signal = hlg_oetf(target_scene)
    return np.clip(hlg_signal, 0.0, 1.0)


def linear_srgb_to_pq_signal(chart_linear_srgb, gamut_matrix, sdr_white_nits=203.0):
    """
    Linear sRGB → target gamut → scale to nits → PQ OETF.
    Returns float64 in [0,1] (PQ signal), shape same as input.
    """
    flat = chart_linear_srgb.reshape(-1, 3)
    target_linear = (gamut_matrix @ flat.T).T
    target_linear = np.maximum(target_linear, 0.0)
    target_nits = target_linear.reshape(chart_linear_srgb.shape) * sdr_white_nits

    pq_signal = pq_oetf(target_nits)
    return np.clip(pq_signal, 0.0, 1.0)


def linear_srgb_to_srgb_signal(chart_linear_srgb):
    """Linear sRGB → sRGB OETF (no gamut change). Float64 [0,1]."""
    return np.clip(srgb_oetf(chart_linear_srgb), 0.0, 1.0)
