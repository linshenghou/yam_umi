"""Export stable YAM model assets for MuJoCo replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model_assets import DEFAULT_MODEL_DIR, export_model_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YAM + LINEAR_4310 + tracker model assets."
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for exported XML, URDF, and metadata.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if any exported asset already exists.",
    )
    args = parser.parse_args()

    assets = export_model_assets(args.out_dir, overwrite=not args.no_overwrite)
    ok = assets.metadata["validation"]["t_grasp_tracker_matches_config"]
    print(f"XML:  {assets.xml_path}")
    print(f"URDF: {assets.urdf_path}")
    print(f"META: {assets.meta_path}")
    print(f"T_grasp_tracker validation: {ok}")


if __name__ == "__main__":
    main()
