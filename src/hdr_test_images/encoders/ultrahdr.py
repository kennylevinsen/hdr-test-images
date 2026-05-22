"""
Ultra HDR (ISO 21496-1 / Google Ultra HDR) encoder via `ultrahdr_app`.

ultrahdr_app v1.4.0 reference (from `ultrahdr_app -h`):
    -m 0                            encode
    -p <raw HDR>                    10-bit raw HDR pixels
    -i <SDR.jpg>                    compressed JPEG SDR base
    -w / -h                         dimensions
    -a 5                            HDR color format = rgba1010102
    -t {1,2}                        HDR transfer: 1=HLG, 2=PQ
    -C 2                            HDR gamut = bt2100 (BT.2020 primaries)
    -c 0                            SDR gamut = bt709 (sRGB)
    -R 1                            HDR is full-range
    -L 1000                         target display peak brightness in nits
    -Q 100                          gainmap JPEG quality
    -z <out.jpg>                    output filename

rgba1010102 layout (Android / DRM ABGR2101010):
    bit 0..9  = R
    bit 10..19 = G
    bit 20..29 = B
    bit 30..31 = A    (we set A=3 = opaque)
Stored as little-endian 32-bit words.

We only encode two variants: HLG-BT.2020-sourced and PQ-BT.2020-sourced. The
SDR base is `test_reference.jpg`, the existing sRGB JPEG already produced
by the pipeline.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..quantize import to_uint


def _pack_rgba1010102(arr_u10):
    """
    (H, W, 3) uint16 with values in [0, 1023] → (H, W) uint32 packed as
    rgba1010102 with A=3. Little-endian 32-bit words.
    """
    if arr_u10.dtype not in (np.uint16, np.int32, np.int64):
        arr_u10 = arr_u10.astype(np.uint16)
    if arr_u10.max() > 1023:
        raise ValueError(f"rgba1010102 value out of range: max {arr_u10.max()}")
    r = arr_u10[..., 0].astype(np.uint32)
    g = arr_u10[..., 1].astype(np.uint32)
    b = arr_u10[..., 2].astype(np.uint32)
    a = np.uint32(3)  # opaque, max alpha for 2-bit
    return (a << 30) | (b << 20) | (g << 10) | r


def encode(signal_f64, sdr_jpeg_path, out_path, transfer, hdr_gamut=2):
    """
    signal_f64: float64 [0,1] HDR signal already in target gamut + transfer.
    transfer: 1 = HLG, 2 = PQ.
    hdr_gamut: 2 = BT.2020/BT.2100, 1 = P3, 0 = BT.709 (default 2).
    """
    if transfer not in (1, 2):
        raise ValueError("transfer must be 1 (HLG) or 2 (PQ)")

    h, w, c = signal_f64.shape
    assert c == 3

    arr_u10 = to_uint(signal_f64, bits=10)
    packed = _pack_rgba1010102(arr_u10)
    raw_bytes = packed.astype("<u4").tobytes()

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        raw_path = Path(f.name)
        f.write(raw_bytes)

    try:
        subprocess.run([
            "ultrahdr_app",
            "-m", "0",
            "-p", str(raw_path),
            "-i", str(sdr_jpeg_path),
            "-w", str(w),
            "-h", str(h),
            "-a", "5",                  # rgba1010102
            "-t", str(transfer),
            "-C", str(hdr_gamut),
            "-c", "0",                  # SDR gamut = bt709 / sRGB
            "-R", "1",                  # HDR full range
            "-L", "1000",               # 1000-nit target
            "-Q", "100",                # gainmap quality
            # Cap content-boost recommendation to 1.0 linear (= no boost).
            # The HDR and SDR inputs encode the same intended display
            # appearance, so the gain map should be ~flat. Without this cap,
            # libultrahdr 1.4.0 picks large default boost ranges that differ
            # between its HLG and PQ encoder paths, so the same intended
            # display appearance produces visibly divergent Ultra HDR JPEGs
            # in Ultra HDR-aware viewers (tev, Chrome 113+). The cap reduces
            # the residual visual difference to a barely-perceptible level
            # but costs round-trip fidelity: HLG reconstruction can drift up
            # to ~40 LSB at 10-bit because the gain map no longer has the
            # headroom to express the encoded HDR signal exactly.
            "-k", "1",
            "-K", "1",
            "-z", str(out_path),
        ], check=True, capture_output=True)
    finally:
        raw_path.unlink(missing_ok=True)
