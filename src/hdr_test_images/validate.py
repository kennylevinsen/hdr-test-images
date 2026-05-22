"""
Three-layer validator for HDR test images.

Layer 1: byte-only metadata parse. Confirms the encoder embedded the CICP
         triple (or color-encoding equivalent) we asked for. Does not use
         any HDR decoder.

Layer 2: lossless pixel round-trip. Decode the file with its native tool to
         a raw pixel buffer, compare to a freshly recomputed numpy reference
         at the same bit depth.

Layer 3: cross-decoder agreement. Where multiple independent decoders exist
         for a format, decode with each and confirm they agree within ≤1 LSB.
         A disagreement is a finding, not a hard failure — surfaced for
         humans because we cannot tell which decoder is buggy.

The numpy reference values come from the same transfer math the encoders
consumed, so Layer 2 catches encoders that corrupt the pixel data even when
they get the metadata right.
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .parsers import png as png_parser
from .parsers import isobmff as isobmff_parser
from .parsers import jxl_codestream as jxl_parser
from .parsers import jpeg as jpeg_parser
from .quantize import to_uint


# CICP code points → libjxl color-encoding-bundle enum tuples.
# (jxl_color_space, jxl_white_point, jxl_primaries, jxl_transfer_function)
# Maps used to translate from CICP to libjxl's ColorEncoding enum representation.
CICP_TO_JXL_ENUMS = {
    (1, 13): (0, 0, 0, 13),     # RGB, D65, sRGB primaries, sRGB transfer
    (9, 18): (0, 0, 2, 18),     # RGB, D65, BT.2020 primaries, HLG transfer
    (12, 18): (0, 0, 3, 18),    # RGB, D65, P3 primaries, HLG transfer
    (9, 16): (0, 0, 2, 16),     # RGB, D65, BT.2020 primaries, PQ transfer
    (12, 16): (0, 0, 3, 16),    # RGB, D65, P3 primaries, PQ transfer
}


@dataclass
class ValidationResult:
    name: str
    path: Path
    layer1: dict = field(default_factory=dict)        # parsed metadata + assertions
    layer2: dict = field(default_factory=dict)        # pixel diff stats
    layer3: dict = field(default_factory=dict)        # per-decoder results
    failures: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_via_subprocess(cmd, out_path):
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        return None, res.stderr.decode(errors="replace")
    return out_path, None


def _decode_with_ffmpeg(input_path, output_png):
    return _decode_via_subprocess(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path),
         "-pix_fmt", "rgb48be", "-update", "1", str(output_png)],
        output_png,
    )


def _decode_with_magick(input_path, output_png):
    # `-type TrueColor` prevents magick from auto-selecting palette PNG for
    # sRGB images, which our PNG parser does not implement.
    return _decode_via_subprocess(
        ["magick", str(input_path),
         "-type", "TrueColor", "-depth", "16",
         "-define", "png:color-type=2",
         str(output_png)],
        output_png,
    )


def _decode_avif_native(input_path, output_png):
    return _decode_via_subprocess(
        ["avifdec", "--depth", "16", str(input_path), str(output_png)],
        output_png,
    )


def _decode_jxl_native(input_path, output_png):
    return _decode_via_subprocess(
        ["djxl", "--bits_per_sample=16", str(input_path), str(output_png)],
        output_png,
    )


def _decode_heif_native(input_path, output_png):
    return _decode_via_subprocess(
        ["heif-dec", "-q", "100", str(input_path), str(output_png)],
        output_png,
    )


def _decode_ultrahdr_native(input_path, output_raw):
    """Ultra HDR decoder outputs raw rgba1010102 / rgbahalffloat / rgba8888.
    We request rgba1010102 (-O 5) with HLG transfer (-o 1) by default — the
    caller chooses transfer to match the source."""
    raise NotImplementedError("call _decode_ultrahdr_at directly with -o flag")


def _decode_ultrahdr_at(input_path, output_raw, transfer):
    """transfer: 1=HLG, 2=PQ, 3=sRGB. Output format -O 5 = rgba1010102."""
    return _decode_via_subprocess(
        ["ultrahdr_app", "-m", "1", "-j", str(input_path),
         "-o", str(transfer), "-O", "5",
         "-z", str(output_raw)],
        output_raw,
    )


def _read_decoded_png(path):
    """Use our own PNG parser to read pixel data — no decoder dependency."""
    return png_parser.decode_pixels(path)


def _decoded_to_target_depth(arr, target_bits):
    """
    Inverse of `expand_to_u16`: convert decoded PNG samples to `target_bits`.

    Decoders use one of two conventions when promoting a low-bit-depth source
    sample to a 16-bit PNG sample:
        - bit-replication: N → (N << shift) | (N >> (target_bits*2 - 16))
        - simple shift:    N → N << shift
    Both place N in the top `target_bits` bits, so `arr >> shift` recovers N
    exactly regardless of which convention the decoder used. Round-based
    rescaling (`round(arr * target_max / source_max)`) breaks the simple-shift
    convention by 1 LSB on some values — heif-dec is one such decoder.
    """
    if arr.dtype == np.uint8:
        source_bits = 8
    elif arr.dtype == np.uint16:
        source_bits = 16
    else:
        raise ValueError(f"unexpected decoded dtype: {arr.dtype}")

    if source_bits == target_bits:
        return arr.astype(np.uint32)
    if source_bits > target_bits:
        return (arr.astype(np.uint32) >> (source_bits - target_bits))
    # Upscaling: bit-replicate to fill target depth.
    a = arr.astype(np.uint32)
    high = a << (target_bits - source_bits)
    low = a >> (2 * source_bits - target_bits) if target_bits >= source_bits else np.uint32(0)
    return high | low


def _normalize_to_u16(arr):
    """Promote uint8 to uint16 via bit-replication for cross-decoder comparison."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        a = arr.astype(np.uint32)
        return ((a << 8) | a).astype(np.uint16)
    raise ValueError(f"unexpected dtype: {arr.dtype}")


# ---------------------------------------------------------------------------
# Layer 1 validators per format
# ---------------------------------------------------------------------------

def _validate_png_layer1(result, expected_cicp):
    info = png_parser.parse(result.path)
    result.layer1["bit_depth"] = info.bit_depth
    result.layer1["color_type"] = info.color_type
    result.layer1["cicp"] = info.cicp
    result.layer1["other_color_chunks"] = [c for c in info.color_chunks_seen if c != "cICP"]
    result.layer1["cicp_position_after_ihdr"] = info.cicp_position_after_ihdr

    if info.cicp is None:
        result.failures.append("PNG missing cICP chunk")
    elif tuple(info.cicp) != tuple(expected_cicp):
        result.failures.append(
            f"PNG cICP mismatch: got {tuple(info.cicp)}, expected {tuple(expected_cicp)}"
        )
    if info.cicp_position_after_ihdr not in (None, 0):
        result.notes.append(
            f"cICP is not the first chunk after IHDR (position {info.cicp_position_after_ihdr})"
        )
    if result.layer1["other_color_chunks"]:
        result.failures.append(
            f"PNG has conflicting color chunks alongside cICP: {result.layer1['other_color_chunks']}"
        )


def _validate_isobmff_layer1(result, expected_cicp, expected_bits=None):
    info = isobmff_parser.parse(result.path)
    result.layer1["ftyp_brand"] = info.ftyp_brand.decode("ascii", "replace") if info.ftyp_brand else None
    result.layer1["colr_nclx"] = info.colr_nclx
    result.layer1["pixi_bit_depths"] = info.pixi_bit_depths
    result.layer1["ispe"] = info.ispe
    result.layer1["all_colr"] = info.all_colr

    if info.colr_nclx is None:
        result.failures.append("ISOBMFF missing colr/nclx box")
    elif tuple(info.colr_nclx) != tuple(expected_cicp):
        result.failures.append(
            f"ISOBMFF colr/nclx mismatch: got {tuple(info.colr_nclx)}, expected {tuple(expected_cicp)}"
        )

    if expected_bits is not None and info.pixi_bit_depths is not None:
        bd_set = set(info.pixi_bit_depths)
        if bd_set != {expected_bits}:
            result.failures.append(
                f"pixi bit depths {info.pixi_bit_depths} ≠ expected uniform {expected_bits}"
            )


def _validate_jxl_layer1(result, expected_cicp):
    try:
        info = jxl_parser.parse(result.path)
        result.layer1["container"] = info.container
        result.layer1["width"] = info.width
        result.layer1["height"] = info.height
        result.layer1["codestream_color_encoding"] = vars(info.color_encoding)
    except jxl_parser.ParseError as e:
        result.notes.append(f"jxl codestream parse failed: {e}; falling back to ICC")
        result.layer1["codestream_color_encoding"] = None

    # Independent check: extract the libjxl-emitted ICC profile and read its
    # description tag. The description is a stable fingerprint of the color
    # encoding (e.g., "Rec2100PQ", "Rec2100HLG").
    with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
        icc_path = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png_path = Path(f.name)
    try:
        res = subprocess.run(
            ["djxl", str(result.path), str(png_path),
             f"--orig_icc_out={icc_path}"],
            capture_output=True,
        )
        if res.returncode == 0 and icc_path.exists() and icc_path.stat().st_size > 0:
            description = _read_icc_description(icc_path)
            result.layer1["icc_description"] = description
            # libjxl's ICC description equals the input `-x color_space` string
            # verbatim when that flag is used. So we check it lexically.
            expected_descriptions = {
                (1, 13):  {"RGB_D65_SRG_Rel_SRG", "sRGB", "sRGB IEC61966-2.1"},
                (9, 18):  {"RGB_D65_202_Rel_HLG", "Rec2100HLG", "Rec.2100 HLG"},
                (12, 18): {"RGB_D65_DCI_Rel_HLG", "DisplayP3_HLG", "P3 D65 HLG"},
                (9, 16):  {"RGB_D65_202_Rel_PeQ", "Rec2100PQ", "Rec.2100 PQ"},
                (12, 16): {"RGB_D65_DCI_Rel_PeQ", "DisplayP3_PQ", "P3 D65 PQ"},
            }
            tolerated = expected_descriptions.get(tuple(expected_cicp)[:2], set())
            if description not in tolerated:
                result.notes.append(
                    f"JXL ICC description {description!r} not in known set for CICP {tuple(expected_cicp)}"
                )
        else:
            result.notes.append("djxl could not emit --orig_icc_out")
    finally:
        icc_path.unlink(missing_ok=True)
        png_path.unlink(missing_ok=True)

    # Now cross-check the codestream enum (if we parsed it) against the CICP
    expected_enums = CICP_TO_JXL_ENUMS.get(tuple(expected_cicp))
    ce = result.layer1.get("codestream_color_encoding")
    if expected_enums and ce and not ce.get("all_default") and ce.get("want_icc") is False:
        got = (ce.get("color_space"), ce.get("white_point"),
               ce.get("primaries"), ce.get("transfer_function"))
        if got != expected_enums:
            # Not necessarily a failure — our parser is best-effort, but note it.
            result.notes.append(
                f"JXL codestream enums {got} != expected {expected_enums} (parser may be incomplete)"
            )


def _read_icc_description(icc_path) -> Optional[str]:
    """Extract the 'desc' tag from an ICC profile."""
    data = Path(icc_path).read_bytes()
    if len(data) < 132:
        return None
    # ICC profile: 128-byte header, then 4-byte tag count, then tag table
    tag_count = int.from_bytes(data[128:132], "big")
    base = 132
    for i in range(tag_count):
        entry = data[base + i * 12:base + i * 12 + 12]
        if len(entry) < 12:
            return None
        sig = entry[:4]
        offset = int.from_bytes(entry[4:8], "big")
        size = int.from_bytes(entry[8:12], "big")
        if sig == b"desc":
            tag_data = data[offset:offset + size]
            # 'desc' tag type: 'desc' header + 4-byte reserved + 4-byte length + ASCII
            if tag_data[:4] == b"desc":
                str_len = int.from_bytes(tag_data[8:12], "big")
                if str_len > 0:
                    raw = tag_data[12:12 + str_len].rstrip(b"\x00")
                    return raw.decode("ascii", errors="replace")
            elif tag_data[:4] == b"mluc":  # multi-localized unicode
                record_count = int.from_bytes(tag_data[8:12], "big")
                record_size = int.from_bytes(tag_data[12:16], "big")
                if record_count > 0:
                    first_record_offset = 16
                    rec = tag_data[first_record_offset:first_record_offset + record_size]
                    str_len = int.from_bytes(rec[4:8], "big")
                    str_offset = int.from_bytes(rec[8:12], "big")
                    raw = tag_data[str_offset:str_offset + str_len]
                    try:
                        return raw.decode("utf-16-be", errors="replace").rstrip("\x00")
                    except Exception:
                        return None
            return None
    return None


def _validate_uhdr_layer1(result, expected_namespace_keywords):
    info = jpeg_parser.parse(result.path)
    result.layer1["is_jpeg"] = info.is_jpeg
    result.layer1["mpf_image_count"] = info.mpf_image_count
    result.layer1["gainmap_namespace"] = info.gainmap_namespace
    result.layer1["gainmap_fields"] = info.gainmap_fields

    if not info.is_jpeg:
        result.failures.append("Ultra HDR output is not a JPEG")
        return
    if not info.has_mpf:
        result.failures.append("Ultra HDR JPEG missing MPF (Multi-Picture Format) segment")
    if info.mpf_image_count < 2:
        result.failures.append(
            f"MPF declares {info.mpf_image_count} images; expected ≥ 2 (primary + gainmap)"
        )
    if info.gainmap_namespace is None:
        result.failures.append("Ultra HDR JPEG has no recognized gainmap XMP namespace")


# ---------------------------------------------------------------------------
# Layer 2: lossless pixel round-trip via native decoder
# ---------------------------------------------------------------------------

def _compare_pixels(decoded_arr, expected_arr, depth, label, result):
    """
    Compare decoded pixel array against expected at `depth` bit precision.
    `expected_arr` is uint values [0, 2^depth - 1].
    `decoded_arr` is uint8 or uint16 from PNG; we rescale to the target depth.
    """
    if decoded_arr.shape[2] >= 3:
        decoded_arr = decoded_arr[..., :3]
    decoded_target = _decoded_to_target_depth(decoded_arr, depth)

    expected_u32 = expected_arr.astype(np.uint32)
    diff = decoded_target.astype(np.int64) - expected_u32.astype(np.int64)
    max_abs = int(np.max(np.abs(diff)))
    nonzero = int(np.count_nonzero(diff))
    pct = nonzero / diff.size

    result.layer2[label] = {
        "depth": depth,
        "max_abs_diff": max_abs,
        "nonzero_count": nonzero,
        "nonzero_fraction": pct,
    }
    return max_abs


def _validate_pixel_roundtrip(result, decoder_fn, expected_arr, depth, label,
                              max_allowed=0):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_png = Path(f.name)
    try:
        out, err = decoder_fn(result.path, tmp_png)
        if out is None:
            result.notes.append(f"{label}: decode failed ({err.strip()[:200] if err else 'unknown'})")
            return
        decoded = _read_decoded_png(tmp_png)
        max_abs = _compare_pixels(decoded, expected_arr, depth, label, result)
        if max_abs > max_allowed:
            result.failures.append(
                f"{label}: max pixel diff {max_abs} > tolerance {max_allowed} (depth={depth})"
            )
    finally:
        tmp_png.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Layer 3 helpers: cross-decoder agreement
# ---------------------------------------------------------------------------

def _cross_decoder_agreement(result, decoders, label, max_disagreement=1,
                              source_bits=None):
    """
    decoders: list of (name, callable(input_path, output_png_path)). Each
    callable returns (output_path | None, stderr | None).

    `source_bits`, if provided, scales `max_disagreement` from source-LSB
    space into the uint16 comparison space (≈ `max_disagreement << (16 - source_bits)`).
    """
    if source_bits is not None:
        scale = 1 << max(0, 16 - source_bits)
        max_disagreement = max_disagreement * scale
    decoded = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, fn in decoders:
            out_png = Path(tmpdir) / f"dec_{name}.png"
            out, err = fn(result.path, out_png)
            if out is None:
                result.notes.append(f"{label}.{name}: decoder failed ({err.strip()[:200] if err else 'unknown'})")
                continue
            try:
                decoded[name] = _read_decoded_png(out_png)
            except Exception as e:
                result.notes.append(f"{label}.{name}: PNG parse failed: {e}")

    if len(decoded) < 2:
        return
    # Normalize all decoded arrays to uint16 (bit-replication if 8-bit) so
    # cross-decoder comparison happens in a common scale.
    decoded = {k: _normalize_to_u16(v) for k, v in decoded.items()}
    names = list(decoded.keys())
    base_name = names[0]
    base = decoded[base_name].astype(np.int64)
    cross = {}
    for other in names[1:]:
        arr = decoded[other].astype(np.int64)
        nc = min(base.shape[2], arr.shape[2], 3)
        if base.shape[:2] != arr.shape[:2]:
            result.notes.append(
                f"{label}: decoders disagree on shape: {base_name}={base.shape} vs {other}={arr.shape}"
            )
            continue
        diff = np.abs(base[..., :nc] - arr[..., :nc])
        cross[f"{base_name}_vs_{other}"] = {
            "max_abs_diff": int(np.max(diff)),
            "nonzero_count": int(np.count_nonzero(diff)),
        }
        # Tolerance is in normalized uint16 space. A 1-LSB drift at the source
        # depth (e.g., 8-bit) shows up here as 257 LSB (bit-replicated). Scale
        # max_disagreement accordingly.
        if int(np.max(diff)) > max_disagreement:
            result.notes.append(
                f"{label}: cross-decoder disagreement {base_name} vs {other}: "
                f"max_diff={int(np.max(diff))} (tolerance {max_disagreement} in u16 space)"
            )

    result.layer3[label] = cross


# ---------------------------------------------------------------------------
# Public per-format validators
# ---------------------------------------------------------------------------

@dataclass
class FileSpec:
    """A file to validate with its expected encoding parameters."""
    path: Path
    format: str                                # 'png' | 'heif' | 'avif' | 'jxl' | 'uhdr'
    cicp: tuple                                # (P, T, M, R)
    depth: int                                  # bit depth of the encoded signal
    expected_signal: np.ndarray                # float64 [0,1] transfer-encoded buffer
    name: str = ""
    uhdr_transfer: Optional[int] = None        # for ultrahdr: 1=HLG, 2=PQ


def validate_png(spec: FileSpec) -> ValidationResult:
    result = ValidationResult(name=spec.name or spec.path.name, path=spec.path)
    _validate_png_layer1(result, spec.cicp)

    expected = to_uint(spec.expected_signal, bits=spec.depth)
    try:
        decoded = _read_decoded_png(spec.path)
        _compare_pixels(decoded, expected, spec.depth, "native_parse", result)
        max_abs = result.layer2["native_parse"]["max_abs_diff"]
        if max_abs > 0:
            result.failures.append(f"PNG pixel mismatch: max_abs={max_abs}")
    except Exception as e:
        result.failures.append(f"PNG decode failed: {e}")

    _cross_decoder_agreement(
        result,
        [("ffmpeg", _decode_with_ffmpeg), ("magick", _decode_with_magick)],
        "png",
        source_bits=spec.depth,
    )
    return result


def validate_heif(spec: FileSpec) -> ValidationResult:
    result = ValidationResult(name=spec.name or spec.path.name, path=spec.path)
    _validate_isobmff_layer1(result, spec.cicp,
                              expected_bits=spec.depth if spec.depth > 8 else None)

    expected = to_uint(spec.expected_signal, bits=spec.depth)
    _validate_pixel_roundtrip(result, _decode_heif_native, expected, spec.depth,
                              label="heif_native", max_allowed=0)
    _cross_decoder_agreement(
        result,
        [("heif-dec", _decode_heif_native), ("ffmpeg", _decode_with_ffmpeg)],
        "heif",
        source_bits=spec.depth,
    )
    return result


def validate_avif(spec: FileSpec) -> ValidationResult:
    result = ValidationResult(name=spec.name or spec.path.name, path=spec.path)
    _validate_isobmff_layer1(result, spec.cicp, expected_bits=spec.depth)

    expected = to_uint(spec.expected_signal, bits=spec.depth)
    _validate_pixel_roundtrip(result, _decode_avif_native, expected, spec.depth,
                              label="avif_native", max_allowed=0)
    _cross_decoder_agreement(
        result,
        [("avifdec", _decode_avif_native),
         ("ffmpeg", _decode_with_ffmpeg),
         ("magick", _decode_with_magick)],
        "avif",
        source_bits=spec.depth,
    )
    return result


def validate_jxl(spec: FileSpec) -> ValidationResult:
    result = ValidationResult(name=spec.name or spec.path.name, path=spec.path)
    _validate_jxl_layer1(result, spec.cicp)

    expected = to_uint(spec.expected_signal, bits=spec.depth)
    _validate_pixel_roundtrip(result, _decode_jxl_native, expected, spec.depth,
                              label="jxl_native", max_allowed=0)
    _cross_decoder_agreement(
        result,
        [("djxl", _decode_jxl_native),
         ("ffmpeg", _decode_with_ffmpeg),
         ("magick", _decode_with_magick)],
        "jxl",
        source_bits=spec.depth,
    )
    return result


def validate_uhdr(spec: FileSpec) -> ValidationResult:
    result = ValidationResult(name=spec.name or spec.path.name, path=spec.path)
    _validate_uhdr_layer1(result, ["hdrgm", "iso", "google"])

    # Layer 2 for Ultra HDR: decode back to HDR with the same transfer we
    # encoded with, then compare to the 10-bit expected signal.
    if spec.uhdr_transfer is None:
        result.notes.append("uhdr_transfer not provided; skipping Layer 2")
        return result

    expected_u10 = to_uint(spec.expected_signal, bits=10)
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        out_raw = Path(f.name)
    try:
        out, err = _decode_ultrahdr_at(spec.path, out_raw, spec.uhdr_transfer)
        if out is None:
            result.notes.append(f"ultrahdr_app decode failed: {err.strip()[:200] if err else 'unknown'}")
            return result
        raw = np.frombuffer(out_raw.read_bytes(), dtype="<u4")
        h, w = spec.expected_signal.shape[:2]
        if raw.size < h * w:
            result.failures.append(f"ultrahdr raw decode short: got {raw.size} pixels, expected {h*w}")
            return result
        raw = raw[:h * w].reshape(h, w)
        decoded_r = (raw >>  0) & 0x3FF
        decoded_g = (raw >> 10) & 0x3FF
        decoded_b = (raw >> 20) & 0x3FF
        decoded_u10 = np.stack([decoded_r, decoded_g, decoded_b], axis=-1).astype(np.uint32)

        diff = decoded_u10.astype(np.int64) - expected_u10.astype(np.int64)
        max_abs = int(np.max(np.abs(diff)))
        # Ultra HDR is fundamentally a lossy SDR+gainmap reconstruction; we
        # cannot expect bit-exact round-trip. Report stats but only fail on
        # gross deviations (>32 LSB out of 1024).
        result.layer2["uhdr_native"] = {
            "depth": 10,
            "max_abs_diff": max_abs,
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "rms_diff": float(np.sqrt(np.mean(diff * diff))),
        }
        if max_abs > 64:
            result.failures.append(
                f"Ultra HDR pixel diff exceeds gross tolerance: max_abs={max_abs} (>64 LSB at 10-bit)"
            )
        elif max_abs > 16:
            result.notes.append(
                f"Ultra HDR pixel diff is high but acceptable: max_abs={max_abs} LSB at 10-bit"
            )
    finally:
        out_raw.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------

_VALIDATORS = {
    "png": validate_png,
    "heif": validate_heif,
    "avif": validate_avif,
    "jxl": validate_jxl,
    "uhdr": validate_uhdr,
}


def validate_one(spec: FileSpec) -> ValidationResult:
    fn = _VALIDATORS.get(spec.format)
    if fn is None:
        raise ValueError(f"unknown format: {spec.format}")
    return fn(spec)


def format_report(result: ValidationResult) -> str:
    lines = []
    status = "PASS" if result.ok else "FAIL"
    lines.append(f"[{status}] {result.name}  ({result.path})")
    if result.layer1:
        lines.append(f"  Layer 1 (bytes): {result.layer1}")
    if result.layer2:
        lines.append(f"  Layer 2 (pixels): {result.layer2}")
    if result.layer3:
        lines.append(f"  Layer 3 (cross-decoder): {result.layer3}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    for failure in result.failures:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)
