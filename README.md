# hdr-test-images

Test images for HEIF HDR images. This is the kind of output you get from modern cameras in their HEIC/HDR/HLG modes. This differs from e.g., iPhone which outputs images in SDR with a separate, optional gain map.

The script generates three images:
1. A reference JPEG
2. A HEIF-encoded image in sRGB
3. A HEIF-encoded image in HLG with BT.2020 primaries meant to match the former two when targeting a display luminance of 1000 nits
4. A HEIF-encoded image in HLG with DCI-P3 primaries (not following BT.2100 recommendations)
5. A HEIF-encoded image in PQ with BT.2020 primaries

The note on display nit value is important. If the color pipeline targets a different value, the script would need to be adjusted to generate a suitable image. However, 1000 nits seem like a common assumption.

You can use [tev](https://github.com/Tom94/tev) to view the examples as they were intended to be rendered.
