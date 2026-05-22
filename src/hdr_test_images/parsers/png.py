"""
PNG parser: walks the chunk stream and extracts cICP, IHDR, and a list of
color-related chunks. Does NOT use libpng / Pillow — pure stdlib.

A "Layer 1" validator: given a PNG file, return the embedded CICP triple as
the encoder wrote it, plus enough state to assert no conflicting color
chunks are also present.
"""

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class PngColorInfo:
    width: int
    height: int
    bit_depth: int
    color_type: int  # 2=RGB, 6=RGBA, 0=Grey, 3=Palette, 4=GreyA
    cicp: Optional[tuple] = None   # (primaries, transfer, matrix, full_range) or None
    color_chunks_seen: list = field(default_factory=list)
    cicp_position_after_ihdr: Optional[int] = None  # 0 = immediately after IHDR


def _decompress_idat(idat_bytes: bytes) -> bytes:
    return zlib.decompress(idat_bytes)


def _unfilter(raw: bytes, width: int, height: int, bytes_per_pixel: int) -> bytes:
    """
    PNG defilter: 5 filter types, applied per scanline.
    Returns the decoded raw pixel bytes (no filter byte).
    """
    stride = width * bytes_per_pixel
    out = bytearray(stride * height)
    prev_row = bytearray(stride)
    pos = 0
    for y in range(height):
        if pos >= len(raw):
            raise ValueError("truncated IDAT")
        filter_type = raw[pos]
        line = bytearray(raw[pos + 1:pos + 1 + stride])
        pos += 1 + stride

        if filter_type == 0:
            pass
        elif filter_type == 1:  # Sub
            for i in range(bytes_per_pixel, stride):
                line[i] = (line[i] + line[i - bytes_per_pixel]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev_row[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = line[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                above = prev_row[i]
                line[i] = (line[i] + (left + above) // 2) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                left = line[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                above = prev_row[i]
                upper_left = prev_row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                p = left + above - upper_left
                pa = abs(p - left)
                pb = abs(p - above)
                pc = abs(p - upper_left)
                if pa <= pb and pa <= pc:
                    pr = left
                elif pb <= pc:
                    pr = above
                else:
                    pr = upper_left
                line[i] = (line[i] + pr) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type: {filter_type}")
        out[y * stride:(y + 1) * stride] = line
        prev_row = line
    return bytes(out)


def parse(path) -> PngColorInfo:
    """Walk chunks, validate CRCs, extract color metadata. Returns PngColorInfo."""
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path}: not a PNG file")

    info = None
    chunks_after_ihdr = 0
    pos = len(PNG_SIGNATURE)

    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        crc_stored = data[pos + 8 + length:pos + 12 + length]
        crc_computed = struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
        if crc_stored != crc_computed:
            raise ValueError(f"{path}: bad CRC on chunk {ctype!r} at offset {pos}")
        pos += 12 + length

        if ctype == b"IHDR":
            w, h, bd, ct, _comp, _filter, _interlace = struct.unpack(">IIBBBBB", payload)
            info = PngColorInfo(width=w, height=h, bit_depth=bd, color_type=ct)
        elif ctype == b"cICP" and info is not None:
            if length != 4:
                raise ValueError(f"{path}: cICP chunk length {length} (expected 4)")
            info.cicp = tuple(payload)
            info.cicp_position_after_ihdr = chunks_after_ihdr
            info.color_chunks_seen.append("cICP")
        elif ctype in (b"sRGB", b"iCCP", b"gAMA", b"cHRM") and info is not None:
            info.color_chunks_seen.append(ctype.decode("ascii"))

        if info is not None and ctype != b"IHDR":
            chunks_after_ihdr += 1

        if ctype == b"IEND":
            break

    if info is None:
        raise ValueError(f"{path}: missing IHDR")
    return info


def decode_pixels(path) -> "np.ndarray":
    """
    Decode pixel data to a numpy array. Supports 8-bit and 16-bit RGB/RGBA.
    Returns (H, W, C) uint8 or uint16 array.
    """
    import numpy as np

    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path}: not a PNG file")

    info = parse(path)
    if info.color_type not in (2, 6):
        raise NotImplementedError(f"PNG color_type {info.color_type} not supported")
    if info.bit_depth not in (8, 16):
        raise NotImplementedError(f"PNG bit_depth {info.bit_depth} not supported")
    channels = 3 if info.color_type == 2 else 4
    bytes_per_sample = info.bit_depth // 8
    bytes_per_pixel = channels * bytes_per_sample

    # Collect IDAT
    pos = len(PNG_SIGNATURE)
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IDAT":
            idat.extend(payload)
        elif ctype == b"IEND":
            break

    raw = _decompress_idat(bytes(idat))
    pixels = _unfilter(raw, info.width, info.height, bytes_per_pixel)

    if bytes_per_sample == 1:
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(info.height, info.width, channels)
    else:
        # 16-bit samples are stored big-endian
        arr = np.frombuffer(pixels, dtype=">u2").reshape(info.height, info.width, channels)
        arr = arr.astype(np.uint16)

    return arr
