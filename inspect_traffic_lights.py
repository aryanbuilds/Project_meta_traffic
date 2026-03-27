"""Inspect MetaDrive traffic-light control APIs at runtime."""

from __future__ import annotations

import argparse
from pprint import pprint

from envs.intersection_env import bootstrap_and_capture, create_intersection_env, debug_traffic_light_obedience


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and debug MetaDrive traffic lights.")
    parser.add_argument("--steps", type=int, default=6, help="Number of debug steps to run.")
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture the bootstrap snapshot instead of running the obedience debug helper.",
    )
    args = parser.parse_args()

    if args.capture:
        info = bootstrap_and_capture()
        pprint(info)
        return

    env = create_intersection_env()
    try:
        report = debug_traffic_light_obedience(env, steps=args.steps)
        pprint(report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
