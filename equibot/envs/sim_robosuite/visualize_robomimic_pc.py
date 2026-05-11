import argparse
import os
from types import SimpleNamespace

import numpy as np

from equibot.envs.sim_robosuite.convert_robomimic import (
    TASK_TO_ENV,
    _load_states_actions,
    _model_xml,
    _restore_demo_xml,
    _set_state,
    _sorted_demos,
)


def _parse_steps(value):
    if value is None:
        return None
    steps = []
    for item in value.split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    return steps


def _sample_pairs(data, demos, num_samples, steps, rng):
    pairs = []
    if steps is not None:
        for demo_name in demos:
            horizon = len(data[demo_name]["states"])
            for step in steps:
                if 0 <= step < horizon:
                    pairs.append((demo_name, step))
                if len(pairs) >= num_samples:
                    return pairs
        return pairs

    for _ in range(num_samples):
        demo_name = demos[int(rng.randint(len(demos)))]
        horizon = len(data[demo_name]["states"])
        pairs.append((demo_name, int(rng.randint(horizon))))
    return pairs


def _equal_axes_3d(ax, points):
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(float(radius), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _save_preview(path, rgb, depth, pc, title, mask=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 4), dpi=140)
    ax_rgb = fig.add_subplot(1, 4, 1)
    ax_mask = fig.add_subplot(1, 4, 2)
    ax_depth = fig.add_subplot(1, 4, 3)
    ax_pc = fig.add_subplot(1, 4, 4, projection="3d")

    ax_rgb.imshow(rgb)
    ax_rgb.set_title("Robomimic state render")
    ax_rgb.axis("off")

    ax_mask.imshow(rgb)
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = mask.astype(np.float32) * 0.55
        ax_mask.imshow(overlay)
    ax_mask.set_title("Selected PC pixels")
    ax_mask.axis("off")

    ax_depth.imshow(depth, cmap="magma")
    ax_depth.set_title("Camera depth")
    ax_depth.axis("off")

    if len(pc) > 0:
        colors = pc[:, 2]
        ax_pc.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, cmap="viridis", s=4)
        _equal_axes_3d(ax_pc, pc)
    ax_pc.set_title(f"EquiBot PC ({len(pc)} pts)")
    ax_pc.set_xlabel("x")
    ax_pc.set_ylabel("y")
    ax_pc.set_zlabel("z")
    ax_pc.view_init(elev=22, azim=-55)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _object_pos_from_env(env):
    obj_pos = env._object_pos_from_obs()
    if obj_pos is None:
        return np.full(3, np.nan, dtype=np.float32)
    return np.asarray(obj_pos, dtype=np.float32)


def visualize(args):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py in the active environment before running this script.") from exc

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    env_args = SimpleNamespace(
        seed=args.seed,
        num_eef=1,
        dof=7,
        max_episode_length=args.max_episode_length,
        freq=args.freq,
        num_points=args.num_points,
        cam_resolution=args.cam_resolution,
        camera_names=args.camera_name,
        pc_crop_radius=args.pc_crop_radius,
        has_renderer=False,
        reward_shaping=False,
        robots=args.robots,
        controller=args.controller,
    )
    env = TASK_TO_ENV[args.task](env_args)

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        demos = _sorted_demos(data)
        if args.num_demos is not None:
            demos = demos[: args.num_demos]
        pairs = _sample_pairs(data, demos, args.num_samples, _parse_steps(args.steps), rng)

        last_demo = None
        for ix, (demo_name, t) in enumerate(pairs):
            demo = data[demo_name]
            states, _ = _load_states_actions(demo)
            if demo_name != last_demo:
                _restore_demo_xml(env, _model_xml(demo))
                last_demo = demo_name

            obs = _set_state(env, states[t])
            render = env.render(return_depth=True, return_pc=True)
            rgb = render["images"][0]
            depth = render["depths"][0]
            pc = render["pc"]
            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            obj_pos = _object_pos_from_env(env)
            pc_mean = pc.mean(axis=0) if len(pc) else np.full(3, np.nan, dtype=np.float32)
            pc_obj_diff = pc_mean - obj_pos
            pc_obj_dist = float(np.linalg.norm(pc_obj_diff)) if np.all(np.isfinite(pc_obj_diff)) else np.nan
            seg = env._render_segmentation(args.camera_name)
            object_ids = env._object_geom_ids()
            mask = np.isin(seg[..., 1], object_ids) if object_ids else None

            stem = f"{args.task}_{demo_name}_t{t:04d}"
            npz_path = os.path.join(args.out_dir, f"{stem}.npz")
            png_path = os.path.join(args.out_dir, f"{stem}.png")
            np.savez(
                npz_path,
                rgb=rgb.astype(np.uint8),
                depth=depth.astype(np.float32),
                pc=pc.astype(np.float32),
                eef_pos=np.asarray(eef_pos, dtype=np.float32),
                object_pos=obj_pos.astype(np.float32),
                pc_mean=pc_mean.astype(np.float32),
                pc_object_diff=pc_obj_diff.astype(np.float32),
                pc_object_dist=np.asarray(pc_obj_dist, dtype=np.float32),
            )
            title = (
                f"{args.task} {demo_name} t={t} | "
                f"obj={obj_pos.round(3)} | "
                f"pc_mean={pc_mean.round(3)} | "
                f"diff={pc_obj_diff.round(3)} | "
                f"dist={pc_obj_dist:.4f}m"
            )
            _save_preview(png_path, rgb, depth, pc, title, mask=mask)
            print(f"saved {png_path} diff={pc_obj_diff.round(4)} dist={pc_obj_dist:.4f}m")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save side-by-side Robomimic state renders and EquiBot point clouds for visual inspection."
    )
    parser.add_argument("--dataset", required=True, help="Path to Robomimic .hdf5 dataset.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_ENV.keys()))
    parser.add_argument("--out_dir", required=True, help="Directory for PNG previews and matching NPZ files.")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--steps", default=None, help="Comma-separated fixed timesteps, e.g. 0,40,80.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--cam_resolution", type=int, default=256)
    parser.add_argument("--camera_name", default="agentview")
    parser.add_argument("--pc_crop_radius", type=float, default=0.45)
    parser.add_argument("--max_episode_length", type=int, default=400)
    parser.add_argument("--freq", type=int, default=20)
    parser.add_argument("--robots", default="Panda")
    parser.add_argument("--controller", default="OSC_POSE")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
