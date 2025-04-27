from isaacsim import SimulationApp
import hydra
import rclpy
import torch
import time
import math
import sys
import numpy as np

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
FILE_PATH = str(Path(__file__).resolve().parent.parent / 'simulation')


@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    # launch omniverse app
    simulation_app = SimulationApp({"headless": False, "anti_aliasing": cfg.sim_app.anti_aliasing,
                                    "width": cfg.sim_app.width, "height": cfg.sim_app.height,
                                    "hide_ui": cfg.sim_app.hide_ui})
    # isaacsim extensions can only be import after app start?
    import omni
    import carb
    from simulation.ros2_bridge.agent_ros2_bridge import RobotDataManager
    import simulation.agent.agent_sensors as agent_sensors
    import simulation.agent.agent_ctrl as agent_ctrl
    from simulation.scene.common import camera_follow
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    print(f'[use cfg] {cfg}')

    if cfg.robot_name == 'go2':
        from simulation.scene.scene_go2 import Go2RSLEnvCfg
        # Go2 Environment setup
        env_cfg = Go2RSLEnvCfg()
        print(f'{cfg.robot_name} env_cfg robot: {env_cfg.scene.legged_robot}')
        # sm = agent_sensors.SensorManagerGo2(cfg.num_envs)
    elif cfg.robot_name == 'a1':
        from simulation.scene.scene_a1 import A1RSLEnvCfg
        # Go2 Environment setup
        env_cfg = A1RSLEnvCfg()
        env_cfg.scene.legged_robot.init_state.pos = tuple(cfg.init_pos)
        env_cfg.scene.legged_robot.init_state.rot = tuple(cfg.init_rot)
        print(f'{cfg.robot_name} env_cfg robot: {env_cfg.scene.legged_robot}')
        sm = agent_sensors.SensorManager(cfg.num_envs, 'A1')
    elif cfg.robot_name == 'aliengo':
        from simulation.scene.scene_aliengo import AliengoRSLEnvCfg
        # Go2 Environment setup
        env_cfg = AliengoRSLEnvCfg()
        env_cfg.scene.legged_robot.init_state.pos = tuple(cfg.init_pos)
        env_cfg.scene.legged_robot.init_state.rot = tuple(cfg.init_rot)
        print(f'{cfg.robot_name} env_cfg robot: {env_cfg.scene.legged_robot}')
        sm = agent_sensors.SensorManager(cfg.num_envs, 'Aliengo')
    else:
        raise NotImplementedError(f'[{cfg.robot_name}] env has not been implemented yet')
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.decimation = math.ceil(1. / env_cfg.sim.dt / cfg.freq)
    print(f'[sim.dt]: {env_cfg.sim.dt}')
    print(f'[decimation]: {env_cfg.decimation}')
    env_cfg.sim.render_interval = env_cfg.decimation
    agent_ctrl.init_base_vel_cmd(cfg.num_envs)
    print(f'[{cfg.robot_name} env_cfg policy]: {env_cfg.observations.policy}')
    print(f'[{cfg.robot_name} env_cfg actuator]: {env_cfg.actions}')

    if cfg.policy == "wmp_loco":
        from locomotion.env_cfg.wmp_env import WMPObsEnvWrapper
        env = ManagerBasedRLEnv(env_cfg)
        env = WMPObsEnvWrapper(env)
        from locomotion.policy.wmp_loco import load_policy_wmp, world_model_data_init, update_wm
        policy = load_policy_wmp(robot_name=cfg.robot_name, device=cfg.policy_device)
        depths = sm.add_camera_wmp(cfg.freq)
        # raise NotImplementedError
    elif cfg.policy == "him_loco":
        from locomotion.env_cfg.him_env import HIMLocoEnvWrapper
        env = ManagerBasedRLEnv(env_cfg)
        env = HIMLocoEnvWrapper(env)
        from locomotion.policy.him_loco import load_policy_him
        policy = load_policy_him(robot_name=cfg.robot_name, device=cfg.policy_device)
    elif cfg.policy == "wtw_loco":
        from locomotion.env_cfg.wtw_env import WTWLocoEnvWrapper
        env_cfg.scene.legged_robot.init_state.joint_pos['.*L_hip_joint'] = 0.0
        env_cfg.scene.legged_robot.init_state.joint_pos['.*R_hip_joint'] = 0.0
        env_cfg.scene.legged_robot.init_state.joint_pos['R[L,R]_thigh_joint'] = 0.8
        env_cfg.observations.policy.base_lin_vel.scale = 2.0
        env_cfg.actions.joint_pos.scale = 0.5
        env = ManagerBasedRLEnv(env_cfg)
        env = WTWLocoEnvWrapper(env)
        from locomotion.policy.wtw_loco import load_policy_wtw
        policy = load_policy_wtw(robot_name=cfg.robot_name, device=cfg.policy_device)
    else:
        raise NotImplementedError(f'Policy {cfg.policy} not implemented')

    # Sensor setup
    agent_sensors.create_view_camera(cfg.view_pos, cfg.view_rot)
    cameras = sm.add_camera(cfg.freq)
    lidar_annotators = sm.add_rtx_lidar()

    # Simulation environment
    if cfg.env_name == "obstacle-dense":
        # from simulation.env.hf_env import create_mixed_terrain_env
        # create_mixed_terrain_env()  # obstacles dense
        from simulation.env.mixed_env import create_mixed_terrain_env, create_mixed_terrain_env2
        create_mixed_terrain_env2()
    elif cfg.env_name == "obstacle-medium":
        from simulation.env.hf_env import create_obstacle_medium_env
        create_obstacle_medium_env()  # obstacles medium
    elif cfg.env_name == "obstacle-sparse":
        from simulation.env.hf_env import create_obstacle_sparse_env
        create_obstacle_sparse_env()  # obstacles sparse
    elif cfg.env_name == "omni":
        from simulation.env.omni_env import create_omni_env
        create_omni_env(cfg.scene_id)
    elif cfg.env_name == "matterport3d":
        from simulation.env.mp3d_env import create_matterport3d_env
        create_matterport3d_env(cfg.scene_id)  # matterport3d
    elif cfg.env_name == "carla":
        from simulation.env.carla_env import create_carla_env
        create_carla_env()

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    system_input.subscribe_to_keyboard_events(keyboard, agent_ctrl.sub_keyboard_event)
    # reset control
    reset_ctrl = agent_ctrl.AgentResetControl()
    system_input.subscribe_to_keyboard_events(keyboard, reset_ctrl.sub_keyboard_event)

    # ROS2 Bridge
    rclpy.init()
    dm = RobotDataManager(env, lidar_annotators, cameras, cfg)

    # Run simulation
    sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    obs, _ = env.reset()
    obs_list = obs.cpu().numpy().tolist()[0]
    obs_list = ["{:.3f}".format(v) for v in obs_list]
    print(f'[init obs shape]: {obs.shape}: {obs_list}')

    if cfg.policy == "wmp_loco":
        # init world_model data
        world_model_data_init(obs)

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)

            obs, _, _, _ = env.step(actions)

            if cfg.policy == "wmp_loco":
                # locomotion use camera input
                depth_tensor = torch.zeros((cfg.num_envs, 64, 64), dtype=torch.float32)
                for i, camera in enumerate(depths):
                    depth = camera.get_depth()
                    if depth is not None:
                        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                        depth = depth / 2 - 0.5
                        depth[(depth <= -0.5) | (depth > 0.5)] = 0.5
                        depth_tensor[i] = torch.from_numpy(depth.copy())
                update_wm(actions, obs, depth_tensor)

            # # ROS2 data
            dm.pub_ros2_data()
            rclpy.spin_once(dm)

            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env, 'legged_robot')

            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt / elapsed_time)
        print(f"\rStep time: {actual_loop_time * 1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)

    dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    run_simulator()
