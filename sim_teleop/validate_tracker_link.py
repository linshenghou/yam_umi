"""Validate how the Vive Tracker mount is represented in the YAM model."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate YAM tracker link/site assumptions."
    )
    parser.add_argument(
        "--urdf-only",
        action="store_true",
        help="Only inspect the YAM URDF for tracker links; skip FK validation.",
    )
    args = parser.parse_args()

    from .robot import RobotBundle, YAM_URDF_PATH
    from .transform import T_EE_TRACK

    print(f"YAM URDF: {YAM_URDF_PATH}")
    root = ET.parse(YAM_URDF_PATH).getroot()
    tracker_links = [
        link.get("name")
        for link in root.iter("link")
        if "tracker" in (link.get("name") or "").lower()
    ]
    tracker_joints = [
        joint.get("name")
        for joint in root.iter("joint")
        if "tracker" in (joint.get("name") or "").lower()
    ]

    if tracker_links or tracker_joints:
        print(f"Tracker links in URDF: {tracker_links}")
        print(f"Tracker joints in URDF: {tracker_joints}")
    else:
        print("No tracker link/joint found in the YAM URDF.")
        print("sim_teleop currently injects a MuJoCo tracker_site for validation.")

    if args.urdf_only:
        return

    bundle = RobotBundle()
    print("FK-derived T_EE_TRACK from injected tracker_site:")
    print(np.array2string(bundle.t_ee_track, precision=6, suppress_small=True))
    print("Configured T_EE_TRACK:")
    print(np.array2string(T_EE_TRACK, precision=6, suppress_small=True))
    print(
        "allclose:",
        bool(np.allclose(bundle.t_ee_track, T_EE_TRACK, atol=1e-5)),
    )


if __name__ == "__main__":
    main()
