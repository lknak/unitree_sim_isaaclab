
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym
import os

from . import pickplace_cylinder_g1_29dof_dex3_joint_env_cfg, waiter_env_cfg


gym.register(
    id="Isaac-PickPlace-Cylinder-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_cylinder_g1_29dof_dex3_joint_env_cfg.PickPlaceG129DEX3JointEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waiter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": waiter_env_cfg.WaiterEnvCfg,
    },
    disable_env_checker=True,
)