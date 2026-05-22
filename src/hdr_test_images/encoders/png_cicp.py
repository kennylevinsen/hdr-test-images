"""
PNG encoder with manual cICP chunk injection.

ffmpeg's PNG encoder does not write the cICP chunk (W3C PNG 3rd Ed., 2024),
so we let ffmpeg produce a vanilla 8- or 16-bit RGB PNG, then splice the
cICP chunk in immediately after IHDR. We also strip any iCCP/sRGB/gAMA/cHRM
chunks ahead of IDAT so they cannot conflict with cICP per spec.

Chunk format (4 bytes payload):
  byte 0: colour primaries (CICP)
  byte 1: transfer function (CICP)
  byte 2: matrix coefficients  (MUST be 0 for PNG = RGB)
  byte 3: video full range flag (0 = narrow, 1 = full)
"""

import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..quantize import TempPng, to_uint, write_png_rgb_depth


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Chunks that conflict with cICP per PNG 3rd Ed.: decoders ignore them
# when cICP is present, but their presence indicates a confused encoder.
CONFLICTING_COLOR_CHUNKS = {b"iCCP", b"sRGB", b"gAMA", b"cHRM"}


def _iter_chunks(data: bytes):
    """Yield (offset, length, type, payload, crc) tuples from a PNG file."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file (signature mismatch)")
    pos = len(PNG_SIGNATURE)
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError(f"truncated chunk header at {pos}")
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        end = pos + 8 + length + 4
        if end > len(data):
            raise ValueError(f"truncated chunk body for {chunk_type!r}")
        payload = data[pos + 8:pos + 8 + length]
        crc = data[pos + 8 + length:end]
        yield (pos, length, chunk_type, payload, crc)
        pos = end
        if chunk_type == b"IEND":
            return


def _build_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """Assemble a PNG chunk: length + type + data + CRC32(type+data)."""
    assert len(chunk_type) == 4
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def inject_cicp(png_bytes: bytes, primaries: int, transfer: int,
                matrix: int = 0, full_range: int = 1) -> bytes:
    """
    Return a new PNG with a cICP chunk inserted directly after IHDR and any
    conflicting color chunks removed. Validates each chunk's CRC on the way
    through so we never propagate corruption.
    """
    if matrix != 0:
        raise ValueError("PNG cICP matrix_coefficients must be 0 (RGB)")
    if full_range not in (0, 1):
        raise ValueError("full_range must be 0 or 1")
    for v, name in ((primaries, "primaries"), (transfer, "transfer")):
        if not (0 <= v <= 255):
            raise ValueError(f"{name} out of byte range: {v}")

    cicp_payload = bytes([primaries, transfer, matrix, full_range])
    cicp_chunk = _build_chunk(b"cICP", cicp_payload)

    out = bytearray(PNG_SIGNATURE)
    ihdr_emitted = False
    saw_idat = False

    for _, length, ctype, payload, crc in _iter_chunks(png_bytes):
        expected_crc = struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
        if crc != expected_crc:
            raise ValueError(f"bad CRC on input chunk {ctype!r}")

        if ctype == b"IHDR":
            out.extend(_build_chunk(ctype, payload))
            out.extend(cicp_chunk)
            ihdr_emitted = True
            continue

        if not saw_idat and ctype in CONFLICTING_COLOR_CHUNKS:
            # drop — would conflict with cICP per spec
            continue

        if ctype == b"IDAT":
            saw_idat = True

        out.extend(_build_chunk(ctype, payload))

    if not ihdr_emitted:
        raise ValueError("input PNG had no IHDR")
    return bytes(out)


def encode(signal_f64, out_path, cicp_primaries, cicp_transfer, bits):
    """
    Encode a transfer-encoded float [0,1] buffer to PNG at the given bit depth
    with a cICP chunk for the given (primaries, transfer). matrix=0, full=1.
    """
    arr_uint = to_uint(signal_f64, bits=bits)

    with TempPng() as tmp_png:
        if bits == 8:
            Image.fromarray(arr_uint, "RGB").save(tmp_png)
        else:
            write_png_rgb_depth(arr_uint, bits=bits, out_path=tmp_png)

        raw = Path(tmp_png).read_bytes()
        patched = inject_cicp(raw, cicp_primaries, cicp_transfer)
        Path(out_path).write_bytes(patched)
