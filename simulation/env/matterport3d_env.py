import os
import gzip, json

from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
# from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.vlnce.utils import ASSETS_DIR

dataset_file_name = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
with gzip.open(dataset_file_name, "rt") as f:
    deserialized_episodes = json.loads(f.read())["episodes"]
task_name = 'go2_matterport_base'

def create_matterport3d_env(episode_idx):
    # 设置环境资产路径
    episode = deserialized_episodes[episode_idx]
    scene_id = episode["scene_id"].split('/')[1]
    print(f"running id/episode: {scene_id}/{episode}")
    # scene_id = "759xd9YjKW5"
    scene_id = "GdvgFV5R1Z5"
    # 定义根原语
    prim = define_prim("/World/Mp3d", "Xform")
    print(f'prim {prim}')
    # 加载你自己的 USD 文件（例如 Office 环境）
    asset_path = os.path.join(ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")
    # 引用 USD 文件
    prim.GetReferences().AddReference(asset_path)
    # 如果你需要在其他对象上添加语义标签（可选）
    # 举例：为某些物体添加语义标签
    # 例如：对于办公室场景中的桌子、椅子、墙壁等
    # office_furniture = get_prim_at_path("/World/Office/Furniture")
    # with office_furniture:
    #     rep.modify.semantics([("class", "furniture")])
    # 如果场景已经加载完毕，可能还需要进一步的场景管理，例如修改渲染设置
    # render_scene()  # 自定义的渲染函数

    # from pxr import UsdPhysics
    init_pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2])
    print(f'mp3d episode init_pos {init_pos}')


def add_physics_properties(collider):
    # 设置物体为刚体
    collider.AddAPI("RigidBody")
    # 设置质量
    collider.GetAPI("RigidBody").mass = 1.0
    # 设置摩擦系数
    collider.GetAPI("Collider").friction = 0.5
    # 根据需要设置其他物理属性


def create_matterport3d_env_task(episode_idx, task_name):
    import numpy as np
    import gymnasium as gym

    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    episode = deserialized_episodes[episode_idx]
    print(f"running episode: {episode}")

    env_cfg = parse_env_cfg(task_name, num_envs=1)
    # if "go2" in task:
    env_cfg.scene.robot.init_state.pos = (
        episode["start_position"][0], episode["start_position"][1], episode["start_position"][2] + 0.4)
    # elif "h1" in args_cli.task:
    #     env_cfg.scene.robot.init_state.pos = (
    #     episode["start_position"][0], episode["start_position"][1], episode["start_position"][2] + 1.0)
    # else:
    #     env_cfg.scene.robot.init_state.pos = (
    #     episode["start_position"][0], episode["start_position"][1], episode["start_position"][2] + 0.5)

    env_cfg.scene.disk_1.init_state.pos = (
    episode["start_position"][0], episode["start_position"][1], episode["start_position"][2] + 2.5)
    env_cfg.scene.disk_2.init_state.pos = (
    episode["reference_path"][-1][0], episode["reference_path"][-1][1], episode["reference_path"][-1][2] + 2.5)
    init_rot = episode["start_rotation"]

    env_cfg.scene.robot.init_state.rot = (init_rot[0], init_rot[1], init_rot[2], init_rot[3])
    # import ipdb; ipdb.set_trace()
    env_cfg.goals = episode["goals"]
    env_cfg.episode_id = episode["episode_id"]
    env_cfg.scene_id = episode["scene_id"].split('/')[1]
    env_cfg.traj_id = episode["trajectory_id"]
    env_cfg.instruction_text = episode["instruction"]["instruction_text"]
    env_cfg.instruction_tokens = episode["instruction"]["instruction_tokens"]
    env_cfg.reference_path = np.array(episode["reference_path"])
    expert_locations = np.array(episode["gt_locations"])
    # expert_locations=expert_locations[:,[0,2,1]]
    # expert_locations[:,1] = -expert_locations[:,1]
    # import ipdb; ipdb.set_trace()
    env_cfg.expert_path = expert_locations
    env_cfg.expert_path_length = len(env_cfg.expert_path)
    env_cfg.expert_time = np.arange(env_cfg.expert_path_length) * 1.0
    print('expert_path', env_cfg.expert_path)

    udf_file = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
    if os.path.exists(udf_file):
        env_cfg.scene.terrain.obj_filepath = udf_file
    else:
        raise ValueError(f"No USD file found in scene directory: {udf_file}")

    env = gym.make(task_name, cfg=env_cfg, render_mode=None)

