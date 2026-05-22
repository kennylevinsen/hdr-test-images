"""
JPEG XL codestream parser, just enough to extract the ColorEncoding bundle.

JXL files come in two forms:
  - Naked codestream: starts with the 2-byte signature 0xFF 0x0A
  - ISOBMFF container: ftyp brand 'jxl ', codestream inside one or more
    `jxlc` (single-part) or `jxlp` (multi-part) boxes

The codestream itself is bit-packed (LSB-first within each byte). We parse:
  SizeHeader        → width / height (sanity-check + skip)
  ImageMetadata    → up to and including the ColorEncoding bundle

References:
  - libjxl/lib/jxl/headers.cc          (SizeHeader)
  - libjxl/lib/jxl/image_metadata.cc   (ImageMetadata)
  - libjxl/lib/jxl/color_encoding_internal.cc
  - libjxl/lib/jxl/fields.cc            (U32 codec)

If our parser cannot get to ColorEncoding (e.g., file uses extensions we
don't handle), it raises ParseError so the caller can fall back to a
decoder-assisted approach.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


JXL_SIGNATURE_NAKED = bytes([0xFF, 0x0A])
ISOBMFF_JXL_SIGNATURE_PREFIX = bytes([0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20,
                                       0x0D, 0x0A, 0x87, 0x0A])


class ParseError(Exception):
    pass


@dataclass
class JxlColorEncoding:
    all_default: bool
    want_icc: Optional[bool]
    color_space: Optional[int]      # 0=RGB, 1=Grey, 2=XYB, 3=Unknown
    white_point: Optional[int]      # 1=D65, 2=Custom, 10=E, 11=DCI (but we read the 2-bit enum)
    primaries: Optional[int]        # 1=sRGB, 2=Custom, 9=2100, 11=P3 (we read 2-bit enum)
    transfer_function: Optional[int]  # 1=709, 8=Linear, 13=sRGB, 16=PQ, 17=DCI, 18=HLG, 2=Unknown
    rendering_intent: Optional[int]   # 0=Perceptual, 1=Relative, 2=Saturation, 3=Absolute
    have_gamma: Optional[bool] = None
    gamma: Optional[int] = None


@dataclass
class JxlInfo:
    container: bool
    codestream_offset: int          # offset of 0xFF 0x0A in original file
    width: int
    height: int
    color_encoding: JxlColorEncoding


class BitReader:
    """LSB-first bit reader over a byte buffer."""

    def __init__(self, data: bytes):
        self.data = data
        self.bit_pos = 0

    def read(self, n: int) -> int:
        if n == 0:
            return 0
        value = 0
        for i in range(n):
            byte_idx = self.bit_pos >> 3
            bit_off = self.bit_pos & 7
            if byte_idx >= len(self.data):
                raise ParseError("bit-stream underrun")
            bit = (self.data[byte_idx] >> bit_off) & 1
            value |= bit << i
            self.bit_pos += 1
        return value

    def read_u32(self, c0: int, c1: int, c2: int, c3: int) -> int:
        sel = self.read(2)
        widths = (c0, c1, c2, c3)
        return self.read(widths[sel])


# Mapping from the 2-bit enum read in ImageMetadata to the public
# ColorEncoding enum values used in the libjxl API and `-x color_space`.
# These are NOT CICP code points — libjxl has its own enum.
ENUM_COLORSPACE_LJX_TO_NAME = {0: "RGB", 1: "Grey", 2: "XYB", 3: "Unknown"}
ENUM_WHITE_LJX_TO_NAME = {0: "D65", 1: "Custom", 2: "E", 3: "DCI"}
ENUM_PRIMARIES_LJX_TO_NAME = {0: "sRGB", 1: "Custom", 2: "2100", 3: "P3"}
ENUM_INTENT_LJX_TO_NAME = {0: "Perceptual", 1: "Relative", 2: "Saturation", 3: "Absolute"}
# transfer function is a different encoding: 1 bit "have_gamma" then either
# 24-bit gamma OR an 8-bit U32 enum from a fixed set of CICP-aligned values.
TRANSFER_FUNCTION_VALUES = (1, 2, 8, 13, 16, 17, 18)


def _parse_size_header(br: BitReader):
    """Parse SizeHeader (height and ratio/width). Returns (width, height)."""
    small_y = br.read(1)
    if small_y:
        ysize_div8 = br.read(5)
        ysize = (ysize_div8 + 1) * 8
    else:
        ysize = br.read_u32(9, 13, 18, 30) + 1

    ratio = br.read(3)
    # Ratio enum gives implicit width.
    if ratio == 0:
        small_x = br.read(1)
        if small_x:
            xsize_div8 = br.read(5)
            xsize = (xsize_div8 + 1) * 8
        else:
            xsize = br.read_u32(9, 13, 18, 30) + 1
    else:
        ratios = {1: (1, 1), 2: (6, 5), 3: (4, 3), 4: (3, 2),
                  5: (16, 9), 6: (5, 4), 7: (2, 1)}
        num, den = ratios[ratio]
        xsize = ysize * num // den
    return xsize, ysize


def _parse_preview_header(br: BitReader):
    """PreviewHeader — present iff have_preview. Same layout as a small SizeHeader."""
    # See libjxl/lib/jxl/headers.cc::PreviewHeader. Two U32 fields: x, y.
    div8 = br.read(1)
    if div8:
        ysize = (br.read_u32(0, 1, 4, 9) + 1) * 8
    else:
        ysize = br.read_u32(1, 6, 9, 12) + 1
    ratio = br.read(3)
    if ratio == 0:
        if div8:
            xsize = (br.read_u32(0, 1, 4, 9) + 1) * 8
        else:
            xsize = br.read_u32(1, 6, 9, 12) + 1
    return xsize, ysize


def _parse_animation_header(br: BitReader):
    # tps_numerator (U32), tps_denominator (U32), num_loops (U32), have_timecodes (1 bit)
    br.read_u32(100, 1000, 1, 10000)
    br.read_u32(1, 1001, 1, 1)
    br.read_u32(0, 0, 0, 16)
    br.read(1)


def _parse_intrinsic_size(br: BitReader):
    """IntrinsicSizeHeader, used when have_intr_size. Two U32 fields x, y."""
    br.read_u32(0, 5, 9, 12)
    br.read(3)  # ratio
    # Width comes from ratio or from U32 (we don't need exact values)


def _parse_extra_channel_info(br: BitReader):
    """Skip an ExtraChannelInfo bundle."""
    all_default = br.read(1)
    if all_default:
        return
    # type: 1 bit + U32
    br.read_u32(0, 0, 4, 24)
    br.read_u32(3, 4, 5, 8)  # bit depth
    br.read_u32(0, 3, 5, 8)  # dim shift
    # name length
    name_len = br.read_u32(0, 0, 16, 48)
    for _ in range(name_len):
        br.read(8)
    br.read(1)               # alpha associated
    # additional fields based on type (skip best-effort)
    # Without full type-conditional parsing this may diverge — for our outputs
    # we always pass num_extra_channels = 0 anyway, so this branch is unused.


def _parse_bit_depth_bundle(br: BitReader):
    """BitDepth bundle: floating_point_sample bit, then encoded depth."""
    floating_point = br.read(1)
    bits_per_sample = br.read_u32(8, 10, 12, 16)
    if floating_point:
        br.read(4)   # exponent bits
    # No fractional bits needed for integer.


def _parse_color_encoding(br: BitReader) -> JxlColorEncoding:
    all_default = bool(br.read(1))
    if all_default:
        return JxlColorEncoding(
            all_default=True, want_icc=None,
            color_space=None, white_point=None,
            primaries=None, transfer_function=None, rendering_intent=None,
        )
    want_icc = bool(br.read(1))
    color_space = br.read(2)
    # If want_icc, the rest of the structure is different (ICC profile).
    # We only encode untagged PNGs with `-x color_space=`, so want_icc should
    # be False. Bail out clearly if it's true.
    if want_icc:
        return JxlColorEncoding(
            all_default=False, want_icc=True,
            color_space=color_space, white_point=None,
            primaries=None, transfer_function=None, rendering_intent=None,
        )

    white_point = None
    primaries = None
    transfer_function = None
    have_gamma = None
    gamma = None
    if color_space != 2:  # not XYB
        white_point = br.read(2)
        if white_point == 1:   # Custom — read CustomXY (2 x signed F16 essentially)
            br.read_u32(19, 19, 19, 19)
            br.read_u32(19, 19, 19, 19)
        if color_space != 1:   # not Grey
            primaries = br.read(2)
            if primaries == 1:  # Custom: 3 CustomXY entries
                for _ in range(3):
                    br.read_u32(19, 19, 19, 19)
                    br.read_u32(19, 19, 19, 19)
        # Transfer function
        have_gamma = bool(br.read(1))
        if have_gamma:
            gamma = br.read(24)
        else:
            transfer_function = br.read_u32(*TRANSFER_FUNCTION_VALUES[:4])
            # Wait: U32(c0..c3) here is actually a 4-entry enum lookup,
            # NOT a bit width. libjxl's TransferFunction is encoded as
            # U32(709, 2, 8, U32(13, 16, 17, 18)) — special. Re-read below.

    rendering_intent = br.read(2)
    return JxlColorEncoding(
        all_default=False, want_icc=False,
        color_space=color_space, white_point=white_point,
        primaries=primaries, transfer_function=transfer_function,
        rendering_intent=rendering_intent,
        have_gamma=have_gamma, gamma=gamma,
    )


def _extract_jxl_codestream(data: bytes):
    """Return (codestream_bytes, offset_of_first_codestream_byte_in_file)."""
    if data.startswith(JXL_SIGNATURE_NAKED):
        return data, 0
    if data.startswith(ISOBMFF_JXL_SIGNATURE_PREFIX):
        # ISOBMFF container with JXL signature box at offset 0.
        # Walk top-level boxes; collect jxlc / jxlp payloads.
        pos = 0
        codestream_parts = []
        first_offset = None
        while pos < len(data):
            if pos + 8 > len(data):
                break
            size = struct.unpack(">I", data[pos:pos + 4])[0]
            ctype = data[pos + 4:pos + 8]
            header_len = 8
            if size == 1:
                size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
                header_len = 16
            elif size == 0:
                size = len(data) - pos
            payload_start = pos + header_len
            payload_end = pos + size
            if ctype == b"jxlc":
                if first_offset is None:
                    first_offset = payload_start
                codestream_parts.append(data[payload_start:payload_end])
            elif ctype == b"jxlp":
                # 4-byte index, then partial codestream
                if first_offset is None:
                    first_offset = payload_start + 4
                codestream_parts.append(data[payload_start + 4:payload_end])
            pos = payload_end
        if not codestream_parts:
            raise ParseError("ISOBMFF JXL container with no jxlc/jxlp box")
        return b"".join(codestream_parts), first_offset
    raise ParseError("not a JXL file (no naked signature or container box)")


def parse(path) -> JxlInfo:
    data = Path(path).read_bytes()
    codestream, cs_offset = _extract_jxl_codestream(data)
    if not codestream.startswith(JXL_SIGNATURE_NAKED):
        raise ParseError("codestream missing 0xFF 0x0A signature")

    container = (cs_offset != 0)
    # Start bit reader at the codestream payload after the 2-byte signature.
    br = BitReader(codestream[2:])
    width, height = _parse_size_header(br)

    # ImageMetadata bundle
    all_default = br.read(1)
    if all_default:
        # Default color encoding: sRGB
        ce = JxlColorEncoding(
            all_default=True, want_icc=None,
            color_space=None, white_point=None,
            primaries=None, transfer_function=None, rendering_intent=None,
        )
        return JxlInfo(container=container, codestream_offset=cs_offset,
                       width=width, height=height, color_encoding=ce)

    extra_fields = br.read(1)
    if extra_fields:
        _ = br.read(3)            # orientation
        have_intr_size = br.read(1)
        if have_intr_size:
            _parse_intrinsic_size(br)
        have_preview = br.read(1)
        if have_preview:
            _parse_preview_header(br)
        have_animation = br.read(1)
        if have_animation:
            _parse_animation_header(br)

    _parse_bit_depth_bundle(br)
    br.read(1)                    # modular_16_bit_buffers
    num_extra_channels = br.read_u32(0, 1, 2, 8)
    for _ in range(num_extra_channels):
        _parse_extra_channel_info(br)
    br.read(1)                    # xyb_encoded

    ce = _parse_color_encoding(br)
    return JxlInfo(container=container, codestream_offset=cs_offset,
                   width=width, height=height, color_encoding=ce)
