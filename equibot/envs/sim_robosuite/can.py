from equibot.envs.sim_robosuite.base import RobosuitePointCloudEnv


class CanEnv(RobosuitePointCloudEnv):
    env_name = "PickPlaceCan"
    object_pos_keys = ("Can_pos", "can_pos", "object_pos")
    object_geom_keywords = ("Can_g0_visual",)
    default_horizon = 400
    default_crop_radius = 0.45
