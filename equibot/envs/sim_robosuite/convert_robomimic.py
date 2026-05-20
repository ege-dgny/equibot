import argparse
import json
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from equibot.envs.sim_robosuite.can import CanEnv
from equibot.envs.sim_robosuite.square import SquareEnv


TASK_TO_ENV = {
    "can": CanEnv,
    "square": SquareEnv,
}


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _sorted_demos(data_group):
    def key(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name

    return sorted(list(data_group.keys()), key=key)


def _model_xml(demo_group):
    if "model_file" not in demo_group.attrs:
        return None
    return _decode_attr(demo_group.attrs["model_file"])


def _load_states_actions(demo_group):
    if "states" not in demo_group:
        raise KeyError("Robomimic demo is missing required dataset 'states'.")
    if "actions" not in demo_group:
        raise KeyError("Robomimic demo is missing required dataset 'actions'.")
    return np.asarray(demo_group["states"]), np.asarray(demo_group["actions"])


def robosuite_to_equibot_action(action, dof=7):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if dof != 7:
        out = np.zeros((dof,), dtype=np.float32)
        out[: min(dof, len(action))] = action[: min(dof, len(action))]
        return out

    out = np.zeros((7,), dtype=np.float32)
    if len(action) >= 7:
        # Robosuite OSC_POSE: xyz, axis-angle, gripper.
        # EquiBot: gripper, xyz, axis-angle.
        out[0] = action[6]
        out[1:4] = action[0:3]
        out[4:7] = action[3:6]
    else:
        out[: len(action)] = action
    return out


def _repair_robosuite_asset_paths(xml):
    """Patch stale robosuite asset paths found in older Robomimic XML demos."""
    root = ET.fromstring(xml)
    asset = root.find("asset")
    if asset is None:
        return xml

    replacements = (
        ("/models/assets/mounts/meshes/", "/models/assets/bases/meshes/"),
        ("/models/assets/mounts/", "/models/assets/bases/"),
    )
    changed = False
    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        path = elem.get("file")
        if path is None or os.path.exists(path):
            continue
        for old, new in replacements:
            if old in path:
                candidate = path.replace(old, new)
                if os.path.exists(candidate):
                    elem.set("file", candidate)
                    changed = True
                    break

    if not changed:
        return xml
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _restore_demo_xml(env, model_xml):
    env.env.reset()
    if model_xml is not None:
        xml = env.env.edit_model_xml(model_xml) if hasattr(env.env, "edit_model_xml") else model_xml
        xml = _repair_robosuite_asset_paths(xml)
        env.env.reset_from_xml_string(xml)
        env.env.sim.reset()


def _set_state(env, state):
    env.env.sim.set_state_from_flattened(state)
    env.env.sim.forward()
    obs = env.env._get_observations(force_update=True)
    env._last_obs = obs
    return obs


def convert_dataset(args):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for Robomimic HDF5 conversion. Install with "
            "`python -m pip install h5py` inside the lfd environment."
        ) from exc

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

    out_dir = os.path.join(args.data_out_dir, "pcs")
    os.makedirs(out_dir, exist_ok=True)
    prefix = args.prefix or os.path.basename(os.path.normpath(args.data_out_dir)) or args.task

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

    saved = 0
    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        demos = _sorted_demos(data)
        if args.num_demos is not None:
            demos = demos[: args.num_demos]

        for ep_ix, demo_name in enumerate(demos):
            demo = data[demo_name]
            states, actions = _load_states_actions(demo)
            if args.skip_model_xml:
                env.env.reset()
            else:
                _restore_demo_xml(env, _model_xml(demo))

            horizon = min(len(states), len(actions))
            if args.max_steps_per_demo is not None:
                horizon = min(horizon, args.max_steps_per_demo)

            for t in range(horizon):
                obs = _set_state(env, states[t])
                render = env.render(return_depth=True, return_pc=True)
                pc = render["pc"]
                if pc is None or len(pc) == 0:
                    if args.skip_empty_pc:
                        continue
                    pc = np.zeros((0, 3), dtype=np.float32)
                action = robosuite_to_equibot_action(actions[t], dof=env.dof)
                state = env._state_from_obs(obs)
                rgb = render["images"][0]

                save_path = os.path.join(
                    out_dir,
                    f"{prefix}_ep{ep_ix:06d}_view0_t{t:04d}.npz",
                )
                np.savez(
                    save_path,
                    pc=pc.astype(np.float32),
                    rgb=rgb.astype(np.uint8),
                    action=action.astype(np.float32),
                    eef_pos=state.astype(np.float32),
                )
                saved += 1

            print(f"Converted {demo_name}: {horizon} steps")

    env.close()
    print(f"Saved {saved} EquiBot timesteps to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Robomimic HDF5 demos to EquiBot point-cloud npz files.")
    parser.add_argument("--dataset", required=True, help="Path to Robomimic .hdf5 dataset.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_ENV.keys()))
    parser.add_argument("--data_out_dir", required=True, help="Output directory; files are written under data_out_dir/pcs.")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--max_steps_per_demo", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--cam_resolution", type=int, default=256)
    parser.add_argument("--camera_name", default="agentview")
    parser.add_argument("--pc_crop_radius", type=float, default=0.45)
    parser.add_argument("--max_episode_length", type=int, default=400)
    parser.add_argument("--freq", type=int, default=20)
    parser.add_argument("--robots", default="Panda")
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--skip_empty_pc", action="store_true")
    parser.add_argument(
        "--skip_model_xml",
        action="store_true",
        help="Do not replay per-demo model XML before restoring states. Useful for older Robomimic XMLs with stale robosuite asset/site names.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert_dataset(parse_args())
