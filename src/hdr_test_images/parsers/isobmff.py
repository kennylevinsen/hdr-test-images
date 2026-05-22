"""
ISOBMFF (MPEG-4 Part 12) box-tree parser, scoped to what we need for HEIF
and AVIF colour-info extraction. Pure stdlib — does not link libheif/libavif.

We walk the box tree looking for:
  - `ftyp` to confirm the brand (heic / mif1 / avif)
  - `colr` with `nclx` to extract (primaries, transfer, matrix, full_range)
  - `pixi` for declared bit depth per channel
  - `ispe` for image spatial extent

A correct file has exactly one `colr` of type `nclx` inside the primary
item's property container (`meta → iprp → ipco`). We accept the first such
box we find; if a file has more than one and they disagree, that's a finding
the caller can surface from the returned `all_colr` list.
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Boxes whose payload is just a sequence of sub-boxes (no extra header).
CONTAINER_BOXES = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl",
    b"udta", b"edts", b"dinf", b"meco",
    b"iprp", b"ipco",
}

# FullBox containers: payload begins with 1 byte version + 3 bytes flags,
# then sub-boxes follow.
FULLBOX_CONTAINERS = {
    b"meta",
}


@dataclass
class Box:
    type: bytes
    size: int
    offset: int           # offset of size field in the file
    payload: bytes        # raw bytes excluding the box header
    children: list = field(default_factory=list)


@dataclass
class IsobmffInfo:
    ftyp_brand: Optional[bytes] = None
    ftyp_compatible: list = field(default_factory=list)
    colr_nclx: Optional[tuple] = None    # (primaries, transfer, matrix, full_range)
    all_colr: list = field(default_factory=list)
    pixi_bit_depths: Optional[list] = None
    ispe: Optional[tuple] = None         # (width, height)


def _parse_box_header(data: bytes, pos: int):
    """Return (size, type, header_len, payload_start, payload_end) or None at EOF."""
    if pos + 8 > len(data):
        return None
    size = struct.unpack(">I", data[pos:pos + 4])[0]
    ctype = data[pos + 4:pos + 8]
    header_len = 8
    if size == 1:
        if pos + 16 > len(data):
            raise ValueError(f"truncated largesize at offset {pos}")
        size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
        header_len = 16
    elif size == 0:
        size = len(data) - pos
    payload_start = pos + header_len
    payload_end = pos + size
    if payload_end > len(data):
        raise ValueError(f"box {ctype!r} at {pos} extends past EOF")
    return size, ctype, header_len, payload_start, payload_end


def _walk(data: bytes, start: int, end: int):
    """Yield top-level boxes within [start, end)."""
    pos = start
    while pos < end:
        h = _parse_box_header(data, pos)
        if h is None:
            return
        size, ctype, hlen, p_start, p_end = h
        payload = data[p_start:p_end]
        yield Box(type=ctype, size=size, offset=pos, payload=payload)
        pos = p_end


def _walk_containers(box: Box, info: IsobmffInfo, data: bytes):
    """Recursively process a box, descending into containers."""
    t = box.type
    if t == b"ftyp":
        if len(box.payload) >= 8:
            info.ftyp_brand = box.payload[:4]
            minor_version_end = 8
            # remaining bytes are compatible_brands (each 4 bytes)
            compat = box.payload[minor_version_end:]
            for i in range(0, len(compat) - 3, 4):
                info.ftyp_compatible.append(compat[i:i + 4])

    elif t in FULLBOX_CONTAINERS:
        # 4 bytes of version + flags, then sub-boxes
        sub_start = 4
        for sub in _walk(box.payload, sub_start, len(box.payload)):
            box.children.append(sub)
            _walk_containers(sub, info, data)

    elif t in CONTAINER_BOXES:
        for sub in _walk(box.payload, 0, len(box.payload)):
            box.children.append(sub)
            _walk_containers(sub, info, data)

    elif t == b"colr":
        # Payload: 4-byte colour_type, then type-specific data.
        if len(box.payload) >= 4:
            colour_type = box.payload[:4]
            if colour_type == b"nclx" and len(box.payload) >= 4 + 7:
                # 4 + (2+2+2+1) = 11 bytes
                primaries = struct.unpack(">H", box.payload[4:6])[0]
                transfer  = struct.unpack(">H", box.payload[6:8])[0]
                matrix    = struct.unpack(">H", box.payload[8:10])[0]
                flags     = box.payload[10]
                full_range = (flags >> 7) & 1
                nclx = (primaries, transfer, matrix, full_range)
                info.all_colr.append(("nclx", nclx))
                if info.colr_nclx is None:
                    info.colr_nclx = nclx
            else:
                info.all_colr.append((colour_type.decode("ascii", "replace"), None))

    elif t == b"pixi":
        # FullBox: 4 bytes version+flags, then 1 byte num_channels, then num_channels bytes
        if len(box.payload) >= 5:
            num_channels = box.payload[4]
            if len(box.payload) >= 5 + num_channels:
                info.pixi_bit_depths = list(box.payload[5:5 + num_channels])

    elif t == b"ispe":
        # FullBox: 4 bytes version+flags, then width (4 bytes), height (4 bytes)
        if len(box.payload) >= 12:
            w, h = struct.unpack(">II", box.payload[4:12])
            info.ispe = (w, h)


def parse(path) -> IsobmffInfo:
    data = Path(path).read_bytes()
    info = IsobmffInfo()
    for box in _walk(data, 0, len(data)):
        _walk_containers(box, info, data)
    return info
