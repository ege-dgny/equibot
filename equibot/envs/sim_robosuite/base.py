import logging
import sys
import types

import numpy as np


def _suppress_sim_import_chatter():
    """Silence gym deprecation notices and robosuite startup logger chatter."""
    gym_notices = types.ModuleType("gym_notices")
    gym_notices_notices = types.ModuleType("gym_notices.notices")
    gym_notices_notices.notices = {}
    gym_notices.notices = gym_notices_notices
    sys.modules.setdefault("gym_notices", gym_notices)
    sys.modules.setdefault("gym_notices.notices", gym_notices_notices)

    logging.getLogger("robosuite_logs").disabled = True
    logging.getLogger("robosuite_logs").setLevel(logging.CRITICAL + 1)


_suppress_sim_import_chatter()

import gym


class RobosuitePointCloudEnv:
    """Robosuite wrapper that exposes EquiBot's pc + proprioception interface."""

    env_name = None
    object_pos_keys = ()
    object_geom_keywords = ()
    camera_names = ("agentview",)
    default_horizon = 400
    default_crop_radius = 0.45

    def __init__(self, args, rng=None):
        self.args = args
        self.rng = rng if rng is not None else np.random.RandomState(args.seed)
        self.num_eef = getattr(args, "num_eef", 1)
        self.dof = getattr(args, "dof", 7)
        self.max_episode_length = getattr(args, "max_episode_length", self.default_horizon)
        self.freq = getattr(args, "freq", 20)
        self.num_points = getattr(args, "num_points", 1024)
        self.camera_height = getattr(args, "camera_height", getattr(args, "cam_resolution", 256))
        self.camera_width = getattr(args, "camera_width", getattr(args, "cam_resolution", 256))
        self.crop_radius = getattr(args, "pc_crop_radius", self.default_crop_radius)
        self.reward_scale = getattr(args, "reward_scale", 1.0)
        self._t = 0
        self._last_reward = 0.0
        self._last_obs = None

        self.env = self._make_env()

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_eef, self.dof), dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_eef, 13), dtype=np.float32)

    def _make_env(self):
        try:
            import os
            os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
            import robosuite as suite
        except ImportError as exc:
            raise ImportError(
                "Robosuite Can/Square wrappers require robosuite in the active environment. "
                "Install it in conda env 'lfd' before running these configs."
            ) from exc

        controller = getattr(self.args, "controller", "OSC_POSE")
        robots = getattr(self.args, "robots", "Panda")
        camera_names = self._as_list(getattr(self.args, "camera_names", self.camera_names))
        controller_config = self._load_controller_config(controller, robots)
        make_kwargs = dict(
            env_name=self.env_name,
            robots=robots,
            has_renderer=getattr(self.args, "has_renderer", False),
            has_offscreen_renderer=True,
            use_camera_obs=False,
            reward_shaping=getattr(self.args, "reward_shaping", False),
            horizon=self.max_episode_length,
            control_freq=self.freq,
            camera_names=camera_names,
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            ignore_done=True,
        )
        if controller_config is not None:
            make_kwargs["controller_configs"] = controller_config
        return suite.make(**make_kwargs)

    def _load_controller_config(self, controller, robots):
        try:
            from robosuite.controllers import load_controller_config
            return load_controller_config(default_controller=controller)
        except ImportError:
            pass
        try:
            from robosuite.controllers import load_composite_controller_config
            robot = robots[0] if isinstance(robots, (list, tuple)) else robots
            return load_composite_controller_config(robot=robot)
        except Exception:
            return None

    def reset(self):
        self._t = 0
        self._last_reward = 0.0
        self._last_obs = self.env.reset()
        return self._state_from_obs(self._last_obs)

    def step(self, action, dummy_reward=False):
        rs_action = self._to_robosuite_action(action)
        self._last_obs, rew, done, info = self.env.step(rs_action)
        self._last_reward = float(rew)
        self._t += 1
        done = bool(done) or self._t >= self.max_episode_length
        reward = 0.0 if dummy_reward else self.compute_reward()
        return self._state_from_obs(self._last_obs), reward, done, info

    def render(self, return_depth=True, return_pc=True, cam_info=None, hide_eef=False):
        camera_name = self._camera_name(cam_info)
        rgb, depth = self._render_rgbd(camera_name)
        out = {"images": [rgb]}
        if return_depth:
            out["depths"] = [depth]
        if return_pc:
            seg = self._render_segmentation(camera_name)
            out["pc"] = self._point_cloud_from_depth(depth, camera_name, seg=seg)
        else:
            out["pc"] = None
        return out

    def compute_reward(self):
        if hasattr(self.env, "_check_success"):
            try:
                return float(self.env._check_success())
            except Exception:
                pass
        return float(self._last_reward) / float(self.reward_scale)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def _camera_name(self, cam_info=None):
        if cam_info is not None and "camera_name" in cam_info:
            return cam_info["camera_name"]
        return self._as_list(getattr(self.args, "camera_names", self.camera_names))[0]

    def _as_list(self, value):
        if isinstance(value, str):
            return [value]
        return list(value)

    def _to_robosuite_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1, self.dof)[0]
        if self.dof == 7:
            # EquiBot layout: grip, xyz velocity, axis-angle velocity.
            # Robosuite OSC_POSE layout: xyz, axis-angle, grip.
            action = np.concatenate([action[1:4], action[4:7], action[:1]], axis=0)
        low, high = self.env.action_spec
        out = np.zeros_like(low, dtype=np.float32)
        n = min(len(out), len(action))
        out[:n] = action[:n]
        return np.clip(out, low, high)

    def _state_from_obs(self, obs):
        eef_pos = self._obs_value(obs, ("robot0_eef_pos", "eef_pos"), 3, default=0.0)
        eef_quat = self._obs_value(obs, ("robot0_eef_quat", "eef_quat"), 4, default=(0.0, 0.0, 0.0, 1.0))
        rot = self._quat_to_mat(eef_quat)
        dir1 = rot[:, 0]
        dir2 = rot[:, 2]
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gripper = self._gripper_state(obs)
        return np.concatenate([eef_pos, dir1, dir2, gravity, [gripper]], axis=0).reshape(1, 13).astype(np.float32)

    def _gripper_state(self, obs):
        val = self._obs_value(obs, ("robot0_gripper_qpos", "gripper_qpos"), None, default=None)
        if val is None:
            return 0.0
        return float(np.mean(val) > 0.0)

    def _object_pos_from_obs(self):
        obs = self._last_obs if self._last_obs is not None else {}
        return self._obs_value(obs, self.object_pos_keys, 3, default=None)

    def _obs_value(self, obs, keys, size, default):
        for key in keys:
            if key in obs:
                arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
                if size is None:
                    return arr
                if arr.size >= size:
                    return arr[:size]
        if default is None:
            return None
        return np.full((size,), default, dtype=np.float32) if np.isscalar(default) else np.asarray(default, dtype=np.float32)

    def _render_rgbd(self, camera_name):
        rendered = self.env.sim.render(
            camera_name=camera_name,
            width=self.camera_width,
            height=self.camera_height,
            depth=True,
        )
        rgb, depth = rendered
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        if rgb.shape[0] == self.camera_height:
            rgb = rgb[::-1]
            depth = depth[::-1]
        return rgb[..., :3].astype(np.uint8), depth.astype(np.float32)

    def _render_segmentation(self, camera_name):
        seg = self.env.sim.render(
            camera_name=camera_name,
            width=self.camera_width,
            height=self.camera_height,
            segmentation=True,
        )
        seg = np.asarray(seg)
        if seg.shape[0] == self.camera_height:
            seg = seg[::-1]
        return seg

    def _point_cloud_from_depth(self, depth, camera_name, seg=None):
        depth = self._real_depth_map(depth)
        intr = self._camera_intrinsic_matrix(camera_name)
        extr = self._camera_extrinsic_matrix(camera_name)
        valid = depth > 0
        if seg is not None:
            object_ids = self._object_geom_ids()
            if len(object_ids) > 0:
                object_mask = np.isin(seg[..., 1], object_ids)
                if np.any(object_mask):
                    valid = valid & object_mask
        ys, xs = np.where(valid)
        if len(xs) == 0:
            ys, xs = np.where(depth > 0)
        z = depth[ys, xs]
        x = (xs - intr[0, 2]) * z / intr[0, 0]
        y = (ys - intr[1, 2]) * z / intr[1, 1]
        cam_points = np.stack([x, y, z, np.ones_like(z)], axis=0)
        points = (extr @ cam_points).T[:, :3]

        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        # Workspace-bounds clamp: drop far-plane/background pixels (depth~1 -> huge
        # world coords) that the object-seg mask occasionally leaks. Without this,
        # render glitches inject garbage points (e.g. x~-265) that wreck the cloud.
        ws = (
            (points[:, 2] > 0.78) & (points[:, 2] < 1.3)
            & (points[:, 0] > -0.5) & (points[:, 0] < 0.8)
            & (points[:, 1] > -0.6) & (points[:, 1] < 0.6)
        )
        points = points[ws]
        points = self._crop_object_points(points)
        if len(points) == 0:
            return points
        if len(points) > self.num_points:
            idx = self.rng.choice(len(points), size=self.num_points, replace=False)
            points = points[idx]
        return points.astype(np.float32)

    def _real_depth_map(self, depth):
        depth = np.asarray(depth, dtype=np.float32)
        if np.max(depth) > 1.0:
            return depth
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1.0 - depth * (1.0 - near / far))

    def _camera_intrinsic_matrix(self, camera_name):
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        fovy = self.env.sim.model.cam_fovy[cam_id]
        f = 0.5 * self.camera_height / np.tan(fovy * np.pi / 360.0)
        return np.array(
            [[f, 0, self.camera_width / 2], [0, f, self.camera_height / 2], [0, 0, 1]],
            dtype=np.float32,
        )

    def _camera_extrinsic_matrix(self, camera_name):
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        camera_pos = self.env.sim.data.cam_xpos[cam_id]
        camera_rot = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = camera_rot
        pose[:3, 3] = camera_pos
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return pose @ camera_axis_correction

    def _object_geom_ids(self):
        ids = []
        keywords = tuple(self.object_geom_keywords)
        if len(keywords) == 0:
            return ids
        for geom_id in range(self.env.sim.model.ngeom):
            name = self.env.sim.model.geom_id2name(geom_id)
            if name is not None and any(key in name for key in keywords):
                ids.append(geom_id)
        return ids

    def _fallback_unproject(self, depth, camera_name):
        depth = np.asarray(depth, dtype=np.float32)
        h, w = depth.shape
        ys, xs = np.where(depth > 0)
        z = depth[ys, xs]
        fov = self._camera_fovy(camera_name)
        fy = 0.5 * h / np.tan(0.5 * fov)
        fx = fy
        x = (xs - w * 0.5) * z / fx
        y = (ys - h * 0.5) * z / fy
        return np.stack([x, -y, z], axis=1)

    def _camera_fovy(self, camera_name):
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        return float(self.env.sim.model.cam_fovy[cam_id]) * np.pi / 180.0

    def _crop_object_points(self, points):
        obj_pos = self._object_pos_from_obs()
        if obj_pos is None:
            return points
        dist = np.linalg.norm(points - obj_pos[None], axis=1)
        cropped = points[dist <= self.crop_radius]
        return cropped if len(cropped) > 0 else points

    def _quat_to_mat(self, quat):
        quat = np.asarray(quat, dtype=np.float32)
        try:
            from robosuite.utils.transform_utils import quat2mat
            return quat2mat(quat).astype(np.float32)
        except Exception:
            x, y, z, w = quat
            return np.array(
                [
                    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
                ],
                dtype=np.float32,
            )
