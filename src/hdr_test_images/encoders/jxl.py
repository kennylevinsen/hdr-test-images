"""
JPEG XL encoder via libjxl's `cjxl`.

Mathematically lossless (-d 0 -e 9) at 16-bit input. Color encoding is set via
`-x color_space=<string>` per libjxl's color_description.cc parser:
   ColorSpace_WhitePoint_Primaries_RenderingIntent_Transfer
The intermediate PNG carries no color metadata so `-x color_space` is
authoritative. For HDR variants we pin `--intensity_target=1000` because
libjxl's heuristic default can otherwise shift midtones.
"""

import subprocess
from pathlib import Path

from ..quantize import TempPng, to_uint, write_png_rgb_depth


# CICP (primaries, transfer) → libjxl color_description string.
CICP_TO_JXL = {
    (1, 13): "RGB_D65_SRG_Rel_SRG",     # sRGB
    (9, 18): "RGB_D65_202_Rel_HLG",     # HLG BT.2020
    (12, 18): "RGB_D65_DCI_Rel_HLG",    # HLG Display P3
    (9, 16): "RGB_D65_202_Rel_PeQ",     # PQ BT.2020
    (12, 16): "RGB_D65_DCI_Rel_PeQ",    # PQ Display P3
}


def encode(signal_f64, out_path, cicp_primaries, cicp_transfer, intensity_target=None):
    """16-bit PNG → lossless JXL with explicit color encoding."""
    key = (cicp_primaries, cicp_transfer)
    if key not in CICP_TO_JXL:
        raise ValueError(f"no JXL color_space mapping for CICP {key}")
    color_space = CICP_TO_JXL[key]

    arr_u16 = to_uint(signal_f64, bits=16)
    with TempPng() as tmp_png:
        write_png_rgb_depth(arr_u16, bits=16, out_path=tmp_png)
        cmd = [
            "cjxl",
            "-d", "0",                                  # mathematically lossless
            "-e", "9",                                  # max effort (smallest file)
            "-x", f"color_space={color_space}",
            str(tmp_png),
            str(out_path),
        ]
        if intensity_target is not None:
            cmd[1:1] = ["--intensity_target", str(int(intensity_target))]
        subprocess.run(cmd, check=True, capture_output=True)
