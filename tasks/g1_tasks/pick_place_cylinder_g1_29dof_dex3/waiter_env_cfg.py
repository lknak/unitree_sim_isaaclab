# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import tempfile
import torch
from dataclasses import MISSING



import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# import public scene configuration
from tasks.common_scene.base_scene_pickplace_cylindercfg import TableCylinderSceneCfg


from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets import Articulation, DeformableObject, RigidObject
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
import numpy as np
from isaaclab.sensors import CameraCfg
from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat, euler_angles_to_quat
##
# Scene definition
##

# CONSTANTS
TABLE_TOP = 0.8
STANLEY_CUP_Z = 0.2 - 0.07
GLASS_Z = 0.05
SCALE_ASSETS = 0.8

Y_OFFSET_TABLE = 0.4

Y_OFFSET_ALL = 0
Z_OFFSET_ALL = 0

# ROBO
ROBO_POS=(0, -0.05 + Y_OFFSET_ALL, 0.8 + Z_OFFSET_ALL)
ROBO_ROT=(0.7071, 0, 0, 0.7071)

BASE_FOLDER_PATH = "/home/lux/Downloads/tray"


def create_rigid(usd_path, prim_path, translation, rotation, scale=[1, 1, 1]):
    return RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(pos=translation, 
                                                      rot=rotation),
            spawn=UsdFileCfg(usd_path=usd_path,
                            scale=scale,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg())
    ) 


@configclass
class ObjectTableSceneCfg(TableCylinderSceneCfg):
    """object table scene configuration class
    
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix()
    # 6. add camera configuration 
    # front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()

    object = create_rigid(f"{BASE_FOLDER_PATH}/stanley_cup_heavy.usd", "{ENV_REGEX_NS}/cup", [-0.3, Y_OFFSET_TABLE + Y_OFFSET_ALL, TABLE_TOP + STANLEY_CUP_Z + Z_OFFSET_ALL], [-2.6999488e-01, -4.6566129e-10, -2.9831426e-10,  9.6286190e-01], scale=(1.0, 1.0, 1.02))

    # Table
    table = create_rigid(f"{BASE_FOLDER_PATH}/Table.usd", "{ENV_REGEX_NS}/table", [0.0, Y_OFFSET_TABLE + 0.1 + Y_OFFSET_ALL, Z_OFFSET_ALL], [1, 0, 0, 0], scale=(1, 1, SCALE_ASSETS))
    tray = create_rigid(f"{BASE_FOLDER_PATH}/tray.usd", "{ENV_REGEX_NS}/tray", [-0.12, Y_OFFSET_TABLE - 0.15 + Y_OFFSET_ALL, TABLE_TOP + Z_OFFSET_ALL], [1, 0, 0, 0], scale=(0.8, 0.8, 2))

    glass1 = create_rigid(f"{BASE_FOLDER_PATH}/trash_bin_small_heavy.usd", "{ENV_REGEX_NS}/glass1", [0, Y_OFFSET_TABLE + Y_OFFSET_ALL, TABLE_TOP + GLASS_Z + Z_OFFSET_ALL], [1, 0, 0, 0], scale=(0.2, 0.2, 0.35))
    glass2 = create_rigid(f"{BASE_FOLDER_PATH}/trash_bin_small_heavy.usd", "{ENV_REGEX_NS}/glass2", [0, Y_OFFSET_TABLE + Y_OFFSET_ALL, TABLE_TOP + GLASS_Z + Z_OFFSET_ALL], [1, 0, 0, 0], scale=(0.2, 0.2, 0.35))
    glass3 = create_rigid(f"{BASE_FOLDER_PATH}/trash_bin_small_heavy.usd", "{ENV_REGEX_NS}/glass3", [0, Y_OFFSET_TABLE + Y_OFFSET_ALL, TABLE_TOP + GLASS_Z + Z_OFFSET_ALL], [1, 0, 0, 0], scale=(0.2, 0.2, 0.35))

    packing_table = None
    packing_table_2 = None
    packing_table_3 = None
    packing_table_4 = None
    packing_table_5 = None
    packing_table_6 = None
    room_walls = None

    robot.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBO_POS,
            rot=ROBO_ROT,
            joint_pos={
                "right_shoulder_roll_joint": -np.pi / 4,
                "left_shoulder_roll_joint": np.pi / 4,
                "right_shoulder_yaw_joint": -np.pi / 8,
                "left_shoulder_yaw_joint": np.pi / 8,
                
                f"right_hand_thumb_1_joint": 30 / 180 * np.pi,
                f"right_hand_thumb_2_joint": -30 / 180 * np.pi,
                f"left_hand_thumb_1_joint": -30 / 180 * np.pi,
                f"left_hand_thumb_2_joint": 30 / 180 * np.pi,
            },
            joint_vel={".*": 0.0},
        ),
    )

    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotPOVCam",
        update_period= 0.02,
        height=160,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.12 + Y_OFFSET_ALL, 1.67675 + Z_OFFSET_ALL), 
                                   rot=(-0.19848, 0.9801, 0.0, 0.0), 
                                   convention="ros"),
    )
    left_wrist_camera.height = 480
    left_wrist_camera.width = 640
    right_wrist_camera.height = 480
    right_wrist_camera.width = 640
    front_camera.height = 480
    front_camera.width = 640

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)



@configclass
class ObservationsCfg:
    """defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class 
        """

        # 1. robot joint state observation
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        # 2. gripper joint state observation 
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)

        # 3. camera image observation
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection
    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    # check if the object is out of the working range
    success = DoneTerm(func=mdp.reset_object_estimate,
                       params={
                           "min_height": 0
                       })# use task completion check function

@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward,weight=1.0)


def reset_object_fun(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    left_right_offset: float = 0.3,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset.write_root_pose_to_sim(
        torch.cat([
            torch.stack(
                [(torch.randint(0, 2, (len(env_ids),)) * 2 - 1) * left_right_offset, 
                 (torch.rand((len(env_ids),)) * 2 - 1) * 0.05 + y_offset + Y_OFFSET_TABLE + Y_OFFSET_ALL, 
                 torch.tensor([TABLE_TOP + STANLEY_CUP_Z + Z_OFFSET_ALL + z_offset] * len(env_ids))], dim=1)
            + env.scene.env_origins[env_ids],
            torch.stack([
                torch.tensor(euler_angles_to_quat(np.array([0, 0, 2 * np.pi * np.random.rand()]))) 
                for _ in range(len(env_ids))], dim=0)], dim=-1)
        .to(torch.float32), env_ids=env_ids)


def reset_glass_fun(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    left_right_offset: float = 0.3,
    left_right_rand: float = 0.1,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("glass1"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset.write_root_pose_to_sim(
        torch.cat([
            torch.stack(
                [(torch.randint(0, 2, (len(env_ids),)) * 2 - 1) * left_right_offset + (torch.rand((len(env_ids),)) * 2 - 1) * left_right_rand, 
                 (torch.rand((len(env_ids),)) * 2 - 1) * 0.05 + y_offset + Y_OFFSET_TABLE + Y_OFFSET_ALL, 
                 torch.tensor([TABLE_TOP + GLASS_Z + Z_OFFSET_ALL + z_offset] * len(env_ids))], dim=1)
            + env.scene.env_origins[env_ids],
            torch.stack([torch.tensor([1, 0, 0, 0]) for _ in range(len(env_ids))], dim=0)], dim=-1)
        .to(torch.float32), env_ids=env_ids)


@configclass
class EventCfg:
    reset_object = EventTermCfg(
        func=reset_object_fun,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "left_right_offset": 0.3,
            "y_offset": 0.1,
            "z_offset": 0.01,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_glass1 = EventTermCfg(
        func=reset_glass_fun,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "left_right_offset": 0.0,
            "left_right_rand": 0.02,
            "y_offset": 0.1,
            "z_offset": 0.01,
            "asset_cfg": SceneEntityCfg("glass1"),
        },
    )
    reset_glass2 = EventTermCfg(
        func=reset_glass_fun,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "left_right_offset": 0.1,
            "left_right_rand": 0.02,
            "y_offset": 0.1,
            "z_offset": 0.01,
            "asset_cfg": SceneEntityCfg("glass2"),
        },
    )
    reset_glass3 = EventTermCfg(
        func=reset_glass_fun,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "left_right_offset": 0.2,
            "left_right_rand": 0.02,
            "y_offset": 0.1,
            "z_offset": 0.01,
            "asset_cfg": SceneEntityCfg("glass3"),
        },
    )


@configclass
class WaiterEnvCfg(ManagerBasedRLEnvCfg):
    """uNITREE G1 robot pick place environment configuration class
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, # environment number: 1
                                                     env_spacing=2.5, # environment spacing: 2.5 meter
                                                     replicate_physics=True # enable physics replication
                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    # MDP settings
    # 3. MDP settings
    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    commands = None # command manager
    rewards: RewardsCfg = RewardsCfg()  # reward manager
    curriculum = None # curriculum manager
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = 3
        # self.sim.physx.bounce_threshold_velocity = 0.1
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        # self.sim.physx.friction_correlation_distance = 0.00625

        # create event manager
        self.event_manager = SimpleEventManager()

        # register "reset object" event
        self.event_manager.register("reset_object_self", SimpleEvent(
            func=lambda env: base_mdp.reset_root_state_uniform(
                env,
                torch.arange(env.num_envs, device=env.device),
                pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
                velocity_range={},
                asset_cfg=SceneEntityCfg("object"),
            )
        ))
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))