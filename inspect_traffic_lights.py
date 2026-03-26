"""Inspect MetaDrive traffic-light control APIs at runtime."""

from pprint import pprint

from envs.intersection_env import bootstrap_and_capture


if __name__ == "__main__":
    info = bootstrap_and_capture()
    pprint(info)
