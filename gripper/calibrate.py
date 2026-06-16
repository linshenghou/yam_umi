"""Interactive BRT encoder → gripper calibration.

Run from the repository root:

    & ".venv\\Scripts\\python.exe" -m gripper.calibrate
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --port COM6
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --open 703 --closed 883   # non-interactive

Workflow (interactive):
    1. Port is auto-detected (override with --port).
    2. You are prompted to move the gripper fully OPEN, then press Enter.
    3. You are prompted to move the gripper fully CLOSED, then press Enter.
    4. Stable samples are captured for each endpoint and saved to
       encoder_calibration.json (repo root).

Direction note: the normalisation map is correct regardless of whether the
OPEN value is larger or smaller than the CLOSED value, so physical ordering
does not matter.
"""
from __future__ import annotations

import argparse
import statistics
import time

from .encoder import (
    CALIBRATION_FILE,
    EncoderCalibration,
    create_instrument,
    find_serial_port,
    read_raw,
)


def _sample_stable(inst, n: int = 10, dt: float = 0.05) -> int | None:
    """Take n samples and return their median. Returns None if all fail."""
    vals = []
    for _ in range(n):
        v = read_raw(inst)
        if v is not None:
            vals.append(v)
        time.sleep(dt)
    if not vals:
        return None
    return int(statistics.median(vals))


def _monitor(inst, seconds: float | None = None) -> None:
    """Print raw values until KeyboardInterrupt (or for `seconds`)."""
    t0 = time.time()
    try:
        while seconds is None or time.time() - t0 < seconds:
            print(f"\rraw = {read_raw(inst)}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()


def _capture(inst, prompt: str) -> int | None:
    """Show live values, wait for Enter, return a stable sample."""
    print(f"\n{prompt}")
    print("(live value shown below — move the gripper now, press Enter to capture)")
    print(">>> ", end="", flush=True)
    # Stream values on the same line until Enter is pressed.
    import threading

    captured: dict[str, int | None] = {"v": None}
    done = threading.Event()

    def reader():
        while not done.is_set():
            captured["v"] = _sample_stable(inst, n=5, dt=0.03)
            print(f"\r>>> raw≈{captured['v']}    ", end="", flush=True)
            time.sleep(0.1)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        input()
    finally:
        done.set()
        th.join(timeout=1.0)
    print()
    return captured["v"]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive BRT encoder → gripper calibration"
    )
    p.add_argument("-p", "--port", default=None,
                   help="Serial port (e.g. COM6). Auto-detect if omitted.")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--slave", type=int, default=1, help="Modbus slave address.")
    p.add_argument("--open", type=int, default=None,
                   help="Set raw_open directly (non-interactive).")
    p.add_argument("--closed", type=int, default=None,
                   help="Set raw_closed directly (non-interactive).")
    p.add_argument("--show", action="store_true",
                   help="Only stream live raw values (Ctrl-C to quit).")
    p.add_argument("--reset", action="store_true",
                   help="Delete the saved calibration and exit.")
    args = p.parse_args()

    # ── Reset ──────────────────────────────────────────────────────────────
    if args.reset:
        if CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()
            print(f"Deleted {CALIBRATION_FILE}")
        else:
            print("No calibration file to delete.")
        return

    # ── Port ───────────────────────────────────────────────────────────────
    port = args.port or find_serial_port()
    if port is None:
        raise SystemExit("ERROR: no serial port found (pass --port COMx)")
    print(f"Using port: {port}")

    inst = create_instrument(port, slave_addr=args.slave, baudrate=args.baudrate)
    try:
        probe = read_raw(inst)
        if probe is None:
            raise SystemExit(
                "ERROR: encoder did not respond. Check wiring/baudrate/slave addr."
            )
        print(f"Encoder OK. current raw = {probe}")

        # ── Show-only mode ─────────────────────────────────────────────────
        if args.show:
            print("Streaming raw values (Ctrl-C to quit)...")
            _monitor(inst)
            return

        # ── Non-interactive set ────────────────────────────────────────────
        if args.open is not None and args.closed is not None:
            cal = EncoderCalibration(raw_open=args.open, raw_closed=args.closed)
            cal.save()
            print(cal)
            return

        # ── Interactive capture ────────────────────────────────────────────
        print("\nCurrent calibration:", EncoderCalibration.load())

        raw_open = args.open
        if raw_open is None:
            raw_open = _capture(inst, "STEP 1: move gripper to FULLY OPEN")
        if raw_open is None:
            raise SystemExit("ERROR: failed to read OPEN value.")

        raw_closed = args.closed
        if raw_closed is None:
            raw_closed = _capture(inst, "STEP 2: move gripper to FULLY CLOSED")
        if raw_closed is None:
            raise SystemExit("ERROR: failed to read CLOSED value.")

        cal = EncoderCalibration(raw_open=raw_open, raw_closed=raw_closed)
        if not cal.is_ready:
            raise SystemExit(
                f"ERROR: open ({raw_open}) == closed ({raw_closed}); "
                "calibration needs two distinct values."
            )
        cal.save()
        print("\nSaved calibration:", cal)

        # ── Verify ────────────────────────────────────────────────────────
        print("\nVerification (move the gripper and watch the normalised value):")
        print("0.0 = closed, 1.0 = open. Ctrl-C to quit.")
        _monitor(inst)
        # after monitor ends, show a final reading
        v = read_raw(inst)
        if v is not None:
            print(f"final: raw={v}  normalised={cal.normalise(v):.3f}")
    finally:
        inst.serial.close()
        print("Port closed.")


if __name__ == "__main__":
    main()
