#!/usr/bin/env python3
"""
Standalone validator: re-validate previously generated test images.

Recomputes the expected transfer-encoded signal in numpy and runs the same
three-layer check as `generate.py`. Useful for verifying files generated
elsewhere (e.g., a different machine's libavif, an older heif-enc build).

Without arguments, validates every file in this directory matching the
`test_*.{heif,png,avif,jxl,jpg}` naming convention.
"""

import argparse
import sys
from pathlib import Path

from hdr_test_images.chart import make_chart
from hdr_test_images.transfers import (
    SRGB_TO_REC2020, SRGB_TO_P3, REC2020_LUMA, P3_LUMA,
    linear_srgb_to_hlg_signal, linear_srgb_to_pq_signal,
    linear_srgb_to_srgb_signal,
)
from hdr_test_images.validate import FileSpec, validate_one, format_report


SRGB_CICP = (1, 13, 0, 1)
HLG_2020  = (9, 18, 0, 1)
HLG_P3    = (12, 18, 0, 1)
PQ_2020   = (9, 16, 0, 1)
PQ_P3     = (12, 16, 0, 1)


def specs_for_directory(dir_path: Path):
    chart = make_chart()
    signal_by_variant = {
        "srgb":   (SRGB_CICP, linear_srgb_to_srgb_signal(chart)),
        "hlg":    (HLG_2020,  linear_srgb_to_hlg_signal(chart, SRGB_TO_REC2020, REC2020_LUMA)),
        "hlg_p3": (HLG_P3,    linear_srgb_to_hlg_signal(chart, SRGB_TO_P3, P3_LUMA)),
        "pq":     (PQ_2020,   linear_srgb_to_pq_signal(chart, SRGB_TO_REC2020)),
        "pq_p3":  (PQ_P3,     linear_srgb_to_pq_signal(chart, SRGB_TO_P3)),
    }
    ext_to_format = {".png": "png", ".heif": "heif", ".avif": "avif", ".jxl": "jxl"}
    default_depth = {"png": {"srgb": 8, "other": 16},
                     "heif": {"srgb": 8, "other": 12},
                     "avif": {"srgb": 12, "other": 12},
                     "jxl": {"srgb": 16, "other": 16}}
    specs = []
    for path in sorted(dir_path.glob("test_*.*")):
        if path.suffix == ".jpg":
            stem = path.stem
            if stem == "test_uhdr_hlg":
                cicp, sig = signal_by_variant["hlg"]
                specs.append(FileSpec(path=path, format="uhdr", cicp=cicp,
                                       depth=10, expected_signal=sig, name=stem,
                                       uhdr_transfer=1))
            elif stem == "test_uhdr_pq":
                cicp, sig = signal_by_variant["pq"]
                specs.append(FileSpec(path=path, format="uhdr", cicp=cicp,
                                       depth=10, expected_signal=sig, name=stem,
                                       uhdr_transfer=2))
            continue
        fmt = ext_to_format.get(path.suffix)
        if fmt is None:
            continue
        # Strip "test_" prefix and the extension to get variant name
        variant = path.stem[len("test_"):]
        if variant not in signal_by_variant:
            continue
        cicp, sig = signal_by_variant[variant]
        depth_table = default_depth[fmt]
        depth = depth_table.get("srgb" if variant == "srgb" else "other")
        specs.append(FileSpec(path=path, format=fmt, cicp=cicp, depth=depth,
                              expected_signal=sig, name=path.name))
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", type=Path,
                   help="Specific files to validate (default: scan current directory)")
    p.add_argument("--dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "img",
                   help="Directory to scan when no paths are given (default: ../img)")
    args = p.parse_args()

    if args.paths:
        all_specs = specs_for_directory(args.dir)
        path_set = {q.resolve() for q in args.paths}
        specs = [s for s in all_specs if s.path.resolve() in path_set]
    else:
        specs = specs_for_directory(args.dir)

    if not specs:
        print("No files matched.", file=sys.stderr)
        sys.exit(1)

    n_pass = n_fail = 0
    for spec in specs:
        result = validate_one(spec)
        print(format_report(result))
        if result.ok:
            n_pass += 1
        else:
            n_fail += 1
    print(f"\nValidation: {n_pass} passed, {n_fail} failed.")
    if n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
