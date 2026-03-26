"""Minimal simulation loop bootstrap.

This loop intentionally starts simple to validate environment + rendering.
"""

import time

from envs.intersection_env import create_intersection_env, render_topdown_frame, save_test_frame


def run(steps: int = 300) -> None:
    env = create_intersection_env()
    try:
        env.reset()
        t0 = time.perf_counter()
        for step in range(steps):
            action = env.action_space.sample()
            env.step(action)
            if step % 30 == 0:
                frame = render_topdown_frame(env)
                save_test_frame(frame, out_path=f"data/frame_{step:04d}.png")
        elapsed = time.perf_counter() - t0
        fps = steps / elapsed if elapsed > 0 else 0.0
        print(f"Simulation complete: steps={steps}, elapsed={elapsed:.2f}s, fps={fps:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    run()
