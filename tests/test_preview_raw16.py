import numpy as np

from preview import stretch_fast_native_to_u8, to_u8_preview


def test_raw16_preview_stretches_before_8bit_quantization() -> None:
    raw = np.full((100, 100), 1000, dtype=np.uint16)
    raw[50, 50] = 1020

    # Both values share the same high byte, which was the old preview path.
    old_u8 = to_u8_preview(raw)
    assert int(old_u8[50, 50]) == int(old_u8[0, 0])

    preview = stretch_fast_native_to_u8(raw, plo=5.0, phi=99.5, sample_stride=1)
    assert int(preview[0, 0]) == 0
    assert int(preview[50, 50]) == 255


def test_gamma_brightens_midtones_without_moving_the_endpoints() -> None:
    raw = np.linspace(1000, 1100, num=101, dtype=np.uint16).reshape(1, -1)

    linear = stretch_fast_native_to_u8(raw, plo=0.0, phi=100.0, sample_stride=1, gamma=1.0)
    brightened = stretch_fast_native_to_u8(raw, plo=0.0, phi=100.0, sample_stride=1, gamma=2.0)

    # gamma=1.0 leaves the endpoints and midtones on the plain linear ramp.
    assert int(linear[0, 0]) == 0
    assert int(linear[0, -1]) == 255
    # gamma>1 keeps both endpoints fixed but lifts every midtone above it.
    assert int(brightened[0, 0]) == 0
    assert int(brightened[0, -1]) == 255
    assert int(brightened[0, 50]) > int(linear[0, 50])
