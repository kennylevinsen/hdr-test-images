"""Test chart definition: 640x320 grayscale ramp + color/gray patch grid."""

import numpy as np

WIDTH, HEIGHT = 640, 320

# (name, linear sRGB rgb tuple)
PATCHES_ROW1 = [
    ("White",     (1.000, 1.000, 1.000)),
    ("18% Gray",  (0.180, 0.180, 0.180)),
    ("50% Gray",  (0.500, 0.500, 0.500)),
    ("Red",       (0.800, 0.100, 0.100)),
    ("Green",     (0.100, 0.600, 0.100)),
    ("Blue",      (0.100, 0.100, 0.800)),
    ("Cyan",      (0.100, 0.600, 0.600)),
    ("Magenta",   (0.600, 0.100, 0.600)),
]

PATCHES_ROW2 = [
    ("Yellow",    (0.600, 0.600, 0.100)),
    ("Skin",      (0.350, 0.200, 0.140)),
    ("5% Gray",   (0.050, 0.050, 0.050)),
    ("10% Gray",  (0.100, 0.100, 0.100)),
    ("25% Gray",  (0.250, 0.250, 0.250)),
    ("75% Gray",  (0.750, 0.750, 0.750)),
    ("90% Gray",  (0.900, 0.900, 0.900)),
    ("Black",     (0.000, 0.000, 0.000)),
]


def make_chart():
    """Build a 640x320 float64 linear-sRGB test chart in [0,1].

    Top half: 16-step grayscale ramp (linear stops).
    Bottom half: two rows of 8 color/gray patches each (80x80 px).
    """
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    # Top band: 16-step ramp (y=0..159)
    for i in range(16):
        x0 = i * (WIDTH // 16)
        x1 = (i + 1) * (WIDTH // 16)
        img[0:160, x0:x1, :] = i / 15.0

    # Middle band: color patches (y=160..319)
    patch_w = WIDTH // 8     # 80px
    patch_h = 80

    for col, (_, rgb) in enumerate(PATCHES_ROW1):
        x0 = col * patch_w
        x1 = (col + 1) * patch_w
        img[160:160 + patch_h, x0:x1, :] = rgb

    for col, (_, rgb) in enumerate(PATCHES_ROW2):
        x0 = col * patch_w
        x1 = (col + 1) * patch_w
        img[160 + patch_h:160 + 2 * patch_h, x0:x1, :] = rgb

    return img
