"""BRT encoder → gripper normalisation.

Reads a BRT Modbus-RTU encoder on a USB-serial port (CH340) and maps the
raw register value to a normalised [0, 1] gripper position.
Calibration (open/closed raw values) is persisted to encoder_calibration.json.
"""
import json
from pathlib import Path

import numpy as np

try:
    import minimalmodbus
    import minimalmodbus as _mm
except ImportError:
    minimalmodbus = None  # type: ignore[assignment]
    _mm = None

try:
    import serial.tools.list_ports as _list_ports
except ImportError:
    _list_ports = None  # type: ignore[assignment]

CALIBRATION_FILE = Path(__file__).parent.parent / "encoder_calibration.json"


def find_serial_port() -> "str | None":
    """Auto-detect a USB-serial port (CH340, CP210x, FTDI, etc.)."""
    if _list_ports is None:
        return None
    ports = _list_ports.comports()
    if not ports:
        return None
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        hwid = (p.hwid or "").lower()
        if any(kw in desc or kw in mfg or kw in hwid
               for kw in ["usb", "ch340", "ch9344", "cp210", "ftdi",
                           "pl2303", "usb-serial"]):
            return p.device
    return ports[0].device


def create_instrument(
    port: str,
    slave_addr: int = 1,
    baudrate: int = 9600,
) -> "minimalmodbus.Instrument":
    """Create a minimalmodbus Instrument for the BRT encoder."""
    if _mm is None:
        raise ImportError("minimalmodbus is not installed")
    inst = _mm.Instrument(port, slave_addr)
    inst.serial.baudrate = baudrate
    inst.serial.bytesize = 8
    inst.serial.parity = _mm.serial.PARITY_NONE
    inst.serial.stopbits = 1
    inst.serial.timeout = 1.0
    inst.mode = _mm.MODE_RTU
    return inst


def read_raw(inst: "minimalmodbus.Instrument") -> "int | None":
    """Read the single-turn register 0x0000. Returns None on failure."""
    try:
        return inst.read_register(0x0000, functioncode=3)
    except Exception:
        return None


class EncoderCalibration:
    """Linear map from raw encoder value to normalised [0, 1] gripper position.

    Convention: 0 = fully closed, 1 = fully open.
    The map is correct regardless of which of raw_open / raw_closed is larger
    (the span carries the sign), so the physical open/closed ordering does not
    need to match the numeric ordering.
    """

    def __init__(
        self,
        raw_closed: "int | None" = None,
        raw_open: "int | None" = None,
    ) -> None:
        self.raw_closed = raw_closed
        self.raw_open = raw_open

    @property
    def is_ready(self) -> bool:
        return (
            self.raw_closed is not None
            and self.raw_open is not None
            and self.raw_open != self.raw_closed
        )

    def normalise(self, raw: int) -> float:
        """Map raw value to [0, 1]. Returns 0.0 if not calibrated."""
        if not self.is_ready:
            return 0.0
        span = self.raw_open - self.raw_closed  # type: ignore[operator]
        return float(np.clip((raw - self.raw_closed) / span, 0.0, 1.0))

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        path.write_text(json.dumps(
            {"raw_closed": self.raw_closed, "raw_open": self.raw_open}, indent=2
        ))
        print(f"Calibration saved → {path}")

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "EncoderCalibration":
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
            return cls(raw_closed=d.get("raw_closed"), raw_open=d.get("raw_open"))
        except Exception:
            return cls()

    def __repr__(self) -> str:
        if self.is_ready:
            return (
                f"EncoderCalibration(closed={self.raw_closed}, "
                f"open={self.raw_open}, span={self.raw_open - self.raw_closed})"  # type: ignore[operator]
            )
        return "EncoderCalibration(NOT READY — record open & closed with O/C)"
