from types import SimpleNamespace
from unittest.mock import patch

from camera_poa import CameraInfo, POACameraDevice
from config import CameraConfig
import camera_poa


def _opened_device() -> POACameraDevice:
    dev = POACameraDevice()
    dev._opened = True
    dev._cam_id = 7
    dev._info = CameraInfo(
        camera_id=7,
        model="test",
        sensor="test",
        is_color=True,
        is_usb3=True,
        max_w=1944,
        max_h=1096,
        bit_depth=12,
        pixel_um=2.9,
        bayer_pattern="RGGB",
        bins=(1,),
        img_formats=("POA_RAW16",),
    )
    return dev


def test_camera_ready_timeout_scales_beyond_long_exposure() -> None:
    assert camera_poa.camera_ready_timeout_s(100.0) == 2.0
    assert camera_poa.camera_ready_timeout_s(3000.0) == 4.75
    assert camera_poa.camera_ready_timeout_s(10000.0) == 13.5


def test_camera_config_clamps_gain_to_sdk_range() -> None:
    dev = _opened_device()
    cfg = CameraConfig(gain=6000)
    ok = camera_poa.pyPOACamera.POAErrors.POA_OK
    gain_attr = SimpleNamespace(minValue=0, maxValue=500)

    with (
        patch.object(camera_poa.pyPOACamera, "StopExposure", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageStartPos", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageSize", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageBin", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageFormat", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "GetImageSize", return_value=(ok, 1944, 1096)),
        patch.object(camera_poa.pyPOACamera, "SetExp", return_value=ok),
        patch.object(
            camera_poa.pyPOACamera,
            "GetConfigAttributesByConfigID",
            return_value=(ok, gain_attr),
        ),
        patch.object(camera_poa.pyPOACamera, "SetGain", return_value=ok) as set_gain,
        patch.object(camera_poa.pyPOACamera, "SetConfig", return_value=ok) as set_config,
        patch.object(camera_poa.pyPOACamera, "GetExp", return_value=(ok, 100000, False)),
        patch.object(camera_poa.pyPOACamera, "GetGain", return_value=(ok, 500, False)),
        patch.object(camera_poa.pyPOACamera, "GetConfig", return_value=(ok, 350, False)),
    ):
        dev.configure(cfg)

    set_gain.assert_called_once_with(7, 500, False)
    set_config.assert_called_once_with(
        7,
        camera_poa.pyPOACamera.POAConfig.POA_OFFSET,
        350,
        False,
    )
    assert cfg.gain == 500
    assert cfg.offset == 350
    assert cfg.exp_ms == 100.0


def test_camera_config_rejects_failed_gain_write() -> None:
    dev = _opened_device()
    cfg = CameraConfig(gain=360)
    ok = camera_poa.pyPOACamera.POAErrors.POA_OK
    gain_attr = SimpleNamespace(minValue=0, maxValue=500)

    with (
        patch.object(camera_poa.pyPOACamera, "StopExposure", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageStartPos", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageSize", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageBin", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "SetImageFormat", return_value=ok),
        patch.object(camera_poa.pyPOACamera, "GetImageSize", return_value=(ok, 1944, 1096)),
        patch.object(camera_poa.pyPOACamera, "SetExp", return_value=ok),
        patch.object(
            camera_poa.pyPOACamera,
            "GetConfigAttributesByConfigID",
            return_value=(ok, gain_attr),
        ),
        patch.object(camera_poa.pyPOACamera, "SetGain", return_value=-1),
        patch.object(camera_poa.pyPOACamera, "SetConfig", return_value=ok),
    ):
        try:
            dev.configure(cfg)
        except RuntimeError as exc:
            assert "SetGain" in str(exc)
        else:
            raise AssertionError("configure must reject a failed SetGain call")
