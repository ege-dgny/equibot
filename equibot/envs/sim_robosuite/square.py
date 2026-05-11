from equibot.envs.sim_robosuite.base import RobosuitePointCloudEnv


class SquareEnv(RobosuitePointCloudEnv):
    env_name = "NutAssemblySquare"
    object_pos_keys = ("SquareNut_pos", "square_pos", "nut_pos", "object_pos")
    object_geom_keywords = ("Square", "square")
    default_horizon = 400
    default_crop_radius = 0.35
