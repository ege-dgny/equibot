import argparse
import os
from types import SimpleNamespace

import numpy as np

from equibot.envs.sim_robosuite.convert_robomimic import (
    TASK_TO_ENV,
    _model_xml,
    _restore_demo_xml,
    _set_state,
)


def _camera_names(env):
    return [env.env.sim.model.camera_id2name(i) for i in range(env.env.sim.model.ncam)]


def _save_grid(path, images, names, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=140)
    axes = np.asarray(axes).reshape(-1)

    for ax, image, name in zip(axes, images, names):
        ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def visualize(args):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py in the active environment before running this script.") from exc

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.makedirs(args.out_dir, exist_ok=True)

    env_args = SimpleNamespace(
        seed=args.seed,
        num_eef=1,
        dof=7,
        max_episode_length=args.max_episode_length,
        freq=args.freq,
        num_points=256,
        cam_resolution=args.cam_resolution,
        camera_names="agentview",
        pc_crop_radius=0.45,
        has_renderer=False,
        reward_shaping=False,
        robots=args.robots,
        controller=args.controller,
    )
    env = TASK_TO_ENV[args.task](env_args)

    with h5py.File(args.dataset, "r") as f:
        demo = f["data"][args.demo]
        states = np.asarray(demo["states"])
        if args.timestep < 0 or args.timestep >= len(states):
            raise ValueError(f"timestep {args.timestep} is outside demo length {len(states)}")

        _restore_demo_xml(env, _model_xml(demo))
        _set_state(env, states[args.timestep])

        names = _camera_names(env)
        images = []
        for name in names:
            rgb, _ = env._render_rgbd(name)
            images.append(rgb)

        stem = f"{args.task}_{args.demo}_t{args.timestep:04d}_all_cameras"
        out_path = os.path.join(args.out_dir, f"{stem}.png")
        _save_grid(out_path, images, names, f"{args.task} {args.demo} t={args.timestep}")
        print(f"saved {out_path}")
        print("cameras:", ", ".join(names))

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Render a restored Robomimic state from all robosuite cameras.")
    parser.add_argument("--dataset", required=True, help="Path to Robomimic .hdf5 dataset.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_ENV.keys()))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--cam_resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_length", type=int, default=400)
    parser.add_argument("--freq", type=int, default=20)
    parser.add_argument("--robots", default="Panda")
    parser.add_argument("--controller", default="OSC_POSE")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
