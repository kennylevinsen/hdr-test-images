"""
JPEG parser scoped to Ultra HDR / gain-map metadata.

Walks JPEG markers, collects APP segments, identifies XMP and MPF data, and
extracts gain-map metadata (HDR capacity, gamma, offset, etc.) so the
validator can confirm libultrahdr wrote the parameters we requested.

Two XMP namespaces appear in the wild for gain-map JPEGs:
  - `http://ns.google.com/photos/1.0/camera/`  (Google Ultra HDR predecessor)
  - `urn:iso:std:iso:21496:-1`                 (ISO 21496-1 final)
We accept either.
"""

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# JPEG markers
SOI = 0xFFD8
EOI = 0xFFD9
APP0 = 0xFFE0
APP1 = 0xFFE1
APP2 = 0xFFE2
APP3 = 0xFFE3


@dataclass
class JpegSegment:
    marker: int
    offset: int
    length: int
    payload: bytes


@dataclass
class UltraHdrInfo:
    is_jpeg: bool = False
    has_mpf: bool = False
    mpf_image_count: int = 0
    xmp_payloads: list = field(default_factory=list)
    gainmap_xmp: Optional[str] = None
    gainmap_namespace: Optional[str] = None
    gainmap_fields: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)


def _walk_segments(data: bytes):
    """Yield (marker, offset, length_including_header, payload_bytes)."""
    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG (missing SOI)")
    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError(f"expected 0xFF marker at offset {pos}")
        # Skip fill bytes
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            return
        marker_byte = data[pos]
        pos += 1
        full_marker = 0xFF00 | marker_byte

        # SOI/EOI/RST markers have no payload
        if full_marker == EOI:
            yield JpegSegment(marker=full_marker, offset=pos - 2, length=2, payload=b"")
            return
        if 0xFFD0 <= full_marker <= 0xFFD9:
            yield JpegSegment(marker=full_marker, offset=pos - 2, length=2, payload=b"")
            continue

        if pos + 2 > len(data):
            return
        length = struct.unpack(">H", data[pos:pos + 2])[0]
        payload = data[pos + 2:pos + length]
        yield JpegSegment(marker=full_marker, offset=pos - 2, length=length + 2, payload=payload)
        pos += length

        # Once we hit SOS (Start of Scan), the next "segment" is entropy-coded
        # data until the next marker (or EOI). For Ultra HDR's primary image
        # we don't need to parse SOS data — the metadata we care about is in
        # APP segments before SOS, and the gainmap image is a separate MPF
        # entry that we locate via byte search.
        if full_marker == 0xFFDA:  # SOS
            # Skip entropy data until next non-RST marker. Use brute scan.
            scan_pos = pos
            while scan_pos < len(data) - 1:
                if data[scan_pos] == 0xFF and data[scan_pos + 1] != 0x00 and not (0xD0 <= data[scan_pos + 1] <= 0xD7):
                    pos = scan_pos
                    break
                scan_pos += 1
            else:
                return


def _extract_xmp(payload: bytes) -> Optional[str]:
    """Return XMP string if this APP1 segment is an XMP payload."""
    # Standard XMP signature
    sig = b"http://ns.adobe.com/xap/1.0/\x00"
    if payload.startswith(sig):
        return payload[len(sig):].decode("utf-8", errors="replace")
    return None


def _parse_mpf(payload: bytes) -> int:
    """Return number of images declared in MPF metadata, or 0 if not MPF."""
    sig = b"MPF\x00"
    if not payload.startswith(sig):
        return 0
    tiff = payload[len(sig):]
    if len(tiff) < 8:
        return 0
    # TIFF header
    byte_order = tiff[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return 0
    magic = struct.unpack(endian + "H", tiff[2:4])[0]
    if magic != 42:
        return 0
    ifd_offset = struct.unpack(endian + "I", tiff[4:8])[0]
    if ifd_offset + 2 > len(tiff):
        return 0
    num_entries = struct.unpack(endian + "H", tiff[ifd_offset:ifd_offset + 2])[0]
    # MPF tag 0xB001 = NumberOfImages
    for i in range(num_entries):
        entry_off = ifd_offset + 2 + i * 12
        if entry_off + 12 > len(tiff):
            break
        tag, dtype, count = struct.unpack(endian + "HHI", tiff[entry_off:entry_off + 8])
        value_off = entry_off + 8
        if tag == 0xB001:  # NumberOfImages
            num_images = struct.unpack(endian + "I", tiff[value_off:value_off + 4])[0]
            return num_images
    return 0


_GAINMAP_NAMESPACES = (
    "http://ns.google.com/photos/1.0/camera/",
    "urn:iso:std:iso:21496:-1",
    "urn:iso:std:iso:ts:21496:-1",       # libultrahdr 1.4.0 binary segment id
    "http://ns.google.com/photos/dd/1.0/device/",
)


# APP2 namespace IDs that signal Ultra HDR / gain-map without using XMP.
_BINARY_GAINMAP_SIGS = (
    b"urn:iso:std:iso:ts:21496:-1\x00",
    b"urn:iso:std:iso:21496:-1\x00",
)


def _detect_gainmap_xmp(xmp_string: str):
    """Return (namespace, fields_dict) if this XMP looks like a gain-map block."""
    for ns in _GAINMAP_NAMESPACES:
        if ns in xmp_string:
            fields = {}
            # Extract attribute-style values (hdrgm:...="..." style)
            for m in re.finditer(r"\b(hdrgm|GContainer|Container|Item|hdr)[:_]([A-Za-z][\w]*)\s*=\s*\"([^\"]+)\"", xmp_string):
                fields[m.group(2)] = m.group(3)
            # Also handle element-style <hdrgm:Field>value</hdrgm:Field>
            for m in re.finditer(r"<(?:hdrgm|hdr|Item):([A-Za-z][\w]*)>([^<]+)</", xmp_string):
                fields.setdefault(m.group(1), m.group(2).strip())
            return ns, fields
    return None, {}


def parse(path) -> UltraHdrInfo:
    data = Path(path).read_bytes()
    info = UltraHdrInfo()
    if not data.startswith(b"\xFF\xD8"):
        return info
    info.is_jpeg = True

    for seg in _walk_segments(data):
        info.segments.append(seg)
        if seg.marker == APP1:
            xmp = _extract_xmp(seg.payload)
            if xmp is not None:
                info.xmp_payloads.append(xmp)
                ns, fields = _detect_gainmap_xmp(xmp)
                if ns and info.gainmap_xmp is None:
                    info.gainmap_xmp = xmp
                    info.gainmap_namespace = ns
                    info.gainmap_fields = fields
        elif seg.marker == APP2:
            num_images = _parse_mpf(seg.payload)
            if num_images > 0:
                info.has_mpf = True
                info.mpf_image_count = num_images
            else:
                # libultrahdr 1.4.0 carries the ISO 21496-1 gain-map metadata
                # as a binary APP2 segment (not XMP), keyed by namespace ID.
                for sig in _BINARY_GAINMAP_SIGS:
                    if seg.payload.startswith(sig):
                        ns = sig.rstrip(b"\x00").decode("ascii")
                        if info.gainmap_namespace is None:
                            info.gainmap_namespace = ns
                            info.gainmap_fields = {"binary_iso_21496_1": True}
                        break

    return info
