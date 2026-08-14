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
