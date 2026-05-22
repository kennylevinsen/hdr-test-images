#!/usr/bin/env python3
"""
Generate HDR test images across multiple formats.

Outputs (23 files):
  test_reference.jpg     — sRGB JPEG reference

  test_srgb.heif         — HEIF sRGB                (CICP 1/13/0/1, 8-bit)
  test_srgb.png          — PNG sRGB w/ cICP         (CICP 1/13/0/1, 8-bit)
  test_srgb.avif         — AVIF sRGB                (CICP 1/13/0/0, 12-bit)
  test_srgb.jxl          — JXL sRGB                  (CICP 1/13/0/1, 16-bit)

  test_hlg.{heif,png,avif,jxl}     — HLG BT.2020   (CICP 9/18/0/1)
  test_hlg_p3.{heif,png,avif,jxl}  — HLG Display P3 (CICP 12/18/0/1)
  test_pq.{heif,png,avif,jxl}      — PQ BT.2020    (CICP 9/16/0/1)
  test_pq_p3.{heif,png,avif,jxl}   — PQ Display P3  (CICP 12/16/0/1)

  test_uhdr_hlg.jpg      — Ultra HDR gain-map JPEG, HLG-sourced
  test_uhdr_pq.jpg       — Ultra HDR gain-map JPEG, PQ-sourced

With a correctly implemented HDR pipeline targeting 1000 nits SDR-white=203,
all variants should look identical.

Dependencies (verified at runtime): heif-enc, ffmpeg, avifenc, cjxl, djxl,
  ultrahdr_app, magick, exiv2 (optional).
Python packages: numpy, PIL.
"""

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from hdr_test_images.chart import make_chart
from hdr_test_images.transfers import (
    SRGB_TO_REC2020, SRGB_TO_P3, REC2020_LUMA, P3_LUMA,
    linear_srgb_to_hlg_signal, linear_srgb_to_pq_signal,
    linear_srgb_to_srgb_signal,
)
from hdr_test_images.encoders import heif as enc_heif
from hdr_test_images.encoders import png_cicp as enc_png
from hdr_test_images.encoders import avif as enc_avif
from hdr_test_images.encoders import jxl as enc_jxl
from hdr_test_images.encoders import ultrahdr as enc_uhdr
from hdr_test_images.validate import (
    FileSpec, validate_one, format_report,
)


# CICP (primaries, transfer, matrix=0, full_range=1)
SRGB_CICP   = (1, 13, 0, 1)
HLG_2020    = (9, 18, 0, 1)
HLG_P3      = (12, 18, 0, 1)
PQ_2020     = (9, 16, 0, 1)
PQ_P3       = (12, 16, 0, 1)


REQUIRED_TOOLS = ["heif-enc", "ffmpeg", "avifenc", "cjxl", "djxl",
                  "ultrahdr_app", "magick"]


def check_tools():
    missing = [t for t in REQUIRED_TOOLS if not shutil.which(t)]
    if missing:
        print(f"ERROR: required tools not on PATH: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


@dataclass
class VariantSignal:
    name: str
    cicp: tuple
    signal: np.ndarray
    formats: list   # which formats to emit


def build_signals(chart):
    """Compute all five transfer-encoded signal buffers, plus their CICP triples."""
    variants = []

    # sRGB
    variants.append(VariantSignal(
        name="srgb",
        cicp=SRGB_CICP,
        signal=linear_srgb_to_srgb_signal(chart),
        formats=["heif", "png", "avif", "jxl"],
    ))

    # HLG BT.2020
    variants.append(VariantSignal(
        name="hlg",
        cicp=HLG_2020,
        signal=linear_srgb_to_hlg_signal(chart, SRGB_TO_REC2020, REC2020_LUMA),
        formats=["heif", "png", "avif", "jxl"],
    ))

    # HLG Display P3
    variants.append(VariantSignal(
        name="hlg_p3",
        cicp=HLG_P3,
        signal=linear_srgb_to_hlg_signal(chart, SRGB_TO_P3, P3_LUMA),
        formats=["heif", "png", "avif", "jxl"],
    ))

    # PQ BT.2020
    variants.append(VariantSignal(
        name="pq",
        cicp=PQ_2020,
        signal=linear_srgb_to_pq_signal(chart, SRGB_TO_REC2020),
        formats=["heif", "png", "avif", "jxl"],
    ))

    # PQ Display P3
    variants.append(VariantSignal(
        name="pq_p3",
        cicp=PQ_P3,
        signal=linear_srgb_to_pq_signal(chart, SRGB_TO_P3),
        formats=["heif", "png", "avif", "jxl"],
    ))

    return variants


def encode_all(out_dir, chart, variants):
    """Encode every (variant, format) pair plus reference JPEG and Ultra HDR pair."""
    written = []

    # Reference JPEG (sRGB) — always written even when --only filters out the
    # sRGB family, because Ultra HDR uses it as the SDR base.
    ref_jpg = out_dir / "test_reference.jpg"
    srgb_signal_for_ref = next((v.signal for v in variants if v.name == "srgb"), None)
    if srgb_signal_for_ref is None:
        srgb_signal_for_ref = linear_srgb_to_srgb_signal(chart)
    srgb_u8 = (np.clip(srgb_signal_for_ref, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(srgb_u8, "RGB").save(ref_jpg, quality=100, subsampling=0)
    print(f"  written: {ref_jpg.name}")
    written.append(("jpeg-ref", ref_jpg, None, None))

    # Per-variant per-format encodes
    for v in variants:
        p, t, m, r = v.cicp
        base_name = f"test_{v.name}"

        if "heif" in v.formats:
            out = out_dir / f"{base_name}.heif"
            if v.name == "srgb":
                enc_heif.encode_srgb(v.signal, out)
                depth = 8
            else:
                enc_heif.encode_hdr(v.signal, out, p, t, bits=12)
                depth = 12
            print(f"  written: {out.name} (HEIF, depth={depth})")
            written.append((f"{v.name}_heif", out, v, depth))

        if "png" in v.formats:
            out = out_dir / f"{base_name}.png"
            bits = 8 if v.name == "srgb" else 16
            enc_png.encode(v.signal, out, p, t, bits=bits)
            print(f"  written: {out.name} (PNG, depth={bits})")
            written.append((f"{v.name}_png", out, v, bits))

        if "avif" in v.formats:
            out = out_dir / f"{base_name}.avif"
            enc_avif.encode(v.signal, out, p, t, bits=12)
            print(f"  written: {out.name} (AVIF, depth=12)")
            written.append((f"{v.name}_avif", out, v, 12))

        if "jxl" in v.formats:
            out = out_dir / f"{base_name}.jxl"
            intensity = 1000 if v.name != "srgb" else None
            enc_jxl.encode(v.signal, out, p, t, intensity_target=intensity)
            print(f"  written: {out.name} (JXL, depth=16)")
            written.append((f"{v.name}_jxl", out, v, 16))

    # Ultra HDR: HLG-source and PQ-source variants (only if their HDR sources were built)
    hlg_signal = next((v.signal for v in variants if v.name == "hlg"), None)
    pq_signal = next((v.signal for v in variants if v.name == "pq"), None)

    if hlg_signal is not None:
        out = out_dir / "test_uhdr_hlg.jpg"
        enc_uhdr.encode(hlg_signal, ref_jpg, out, transfer=1)
        print(f"  written: {out.name} (Ultra HDR, HLG-source)")
        written.append(("uhdr_hlg", out, None, 10))

    if pq_signal is not None:
        out = out_dir / "test_uhdr_pq.jpg"
        enc_uhdr.encode(pq_signal, ref_jpg, out, transfer=2)
        print(f"  written: {out.name} (Ultra HDR, PQ-source)")
        written.append(("uhdr_pq", out, None, 10))

    return written


def build_validation_specs(written, variants):
    """Build FileSpec records for everything that should be validated."""
    specs = []
    for name, path, variant, depth in written:
        if name == "jpeg-ref":
            continue
        if name == "uhdr_hlg":
            sig = next(v.signal for v in variants if v.name == "hlg")
            specs.append(FileSpec(
                path=path, format="uhdr", cicp=HLG_2020,
                depth=10, expected_signal=sig, name=name, uhdr_transfer=1,
            ))
        elif name == "uhdr_pq":
            sig = next(v.signal for v in variants if v.name == "pq")
            specs.append(FileSpec(
                path=path, format="uhdr", cicp=PQ_2020,
                depth=10, expected_signal=sig, name=name, uhdr_transfer=2,
            ))
        else:
            fmt = name.rsplit("_", 1)[-1]
            specs.append(FileSpec(
                path=path, format=fmt, cicp=variant.cicp,
                depth=depth, expected_signal=variant.signal, name=name,
            ))
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "img",
                        help="Output directory for generated files (default: ../img)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation after encoding")
    parser.add_argument("--only", action="append", default=None,
                        help="Only encode these variant names (repeatable)")
    args = parser.parse_args()

    check_tools()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building test chart...")
    chart = make_chart()

    print(f"Computing transfer-encoded signals...")
    variants = build_signals(chart)
    if args.only:
        variants = [v for v in variants if v.name in args.only]
        if not variants:
            print(f"ERROR: --only filtered out all variants", file=sys.stderr)
            sys.exit(1)

    print(f"Encoding outputs to {out_dir}...")
    written = encode_all(out_dir, chart, variants)
    print(f"  total: {len(written)} files")

    if args.no_validate:
        return

    print("\nValidating outputs...")
    specs = build_validation_specs(written, variants)
    n_pass = 0
    n_fail = 0
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
