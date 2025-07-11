from isaacsim import SimulationApp
import hydra
import rclpy
import torch
import time
import math
import sys
import numpy as np
import omni
import carb

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
FILE_PATH = str(Path(__file__).resolve().parent.parent / 'simulation')


@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    # launch omniverse app
    simulation_app = SimulationApp({"headless": False, "anti_aliasing": cfg.sim_app.anti_aliasing,
                                    "width": cfg.sim_app.width, "height": cfg.sim_app.height,
                                    "hide_ui": cfg.sim_app.hide_ui})
    # for not setting
    settings_interface = carb.settings.get_settings()
    settings_interface.set("/persistent/isaac/asset_root/cloud",
                           "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5")
    from simulation.ros2_bridge.agent_ros2_bridge import RobotDataManager
    import simulation.agent.agent_sensors as agent_sensors
    import simulation.agent.agent_ctrl as agent_ctrl
    from simulation.scene.common import camera_follow
    from isaaclab.envs import ManagerBasedRLEnv
    print(f'[use cfg] {cfg}')

    # ===============================================================================================
    # robot env setup
    if cfg.robot_name == 'go2':
        from simulation.scene.scene_go2 import Go2RSLEnvCfg
        env_cfg = Go2RSLEnvCfg()
        print(f'{cfg.robot_name} env_cfg robot: {env_cfg.scene.legged_robot}')
        # sm = agent_sensors.SensorManagerGo2(cfg.num_envs)
    elif cfg.robot_name == 'a1':
        from simulation.scene.scene_a1 import A1RSLEnvCfg
        env_cfg = A1RSLEnvCfg()
        env_cfg.scene.legged_robot.init_state.pos = tuple(cfg.init_pos)
        env_cfg.scene.legged_robot.init_state.rot = tuple(cfg.init_rot)
        sm = agent_sensors.SensorManager(cfg.num_envs, 'A1')
    elif cfg.robot_name == 'aliengo':
        if cfg.policy == "legged_loco":
            from simulation.scene.scene_aliengo_leggedLoco import AliengoLeggedEnvCfg
            env_cfg = AliengoLeggedEnvCfg()
            env_cfg.scene.legged_robot.init_state.pos = tuple(cfg.init_pos)
            env_cfg.scene.legged_robot.init_state.rot = tuple(cfg.init_rot)
            sm = agent_sensors.SensorManager(cfg.num_envs, 'Aliengo')
        else:
            from simulation.scene.scene_aliengo import AliengoRSLEnvCfg
            env_cfg = AliengoRSLEnvCfg()
            env_cfg.scene.legged_robot.init_state.pos = tuple(cfg.init_pos)
            env_cfg.scene.legged_robot.init_state.rot = tuple(cfg.init_rot)
            # env_cfg.scene.height_scanner = None
            # env_cfg.observations.policy.height_scan = None
            sm = agent_sensors.SensorManager(cfg.num_envs, 'Aliengo')
    else:
        raise NotImplementedError(f'[{cfg.robot_name}] env has not been implemented yet')
    print(f'{cfg.robot_name} env_cfg robot: {env_cfg.scene.legged_robot}')
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.decimation = math.ceil(1. / env_cfg.sim.dt / cfg.camera_freq)
    print(f'[sim.dt]: {env_cfg.sim.dt}')
    print(f'[decimation]: {env_cfg.decimation}')
    env_cfg.sim.render_interval = env_cfg.decimation
    agent_ctrl.init_base_vel_cmd(cfg.num_envs)
    print(f'[{cfg.robot_name} env_cfg policy]: {env_cfg.observations.policy}')
    print(f'[{cfg.robot_name} env_cfg actuator]: {env_cfg.actions}')

    # ===============================================================================================
    # Simulation environment
    if cfg.env_name == "tunnel":
        from simulation.terrain.hf_env import tunnel_terrain_single
        env_cfg.scene.terrain = tunnel_terrain_single
    if cfg.env_name == "stairs":
        from simulation.terrain.hf_env import stairs_terrain
        env_cfg.scene.terrain = stairs_terrain
    elif cfg.env_name == "obstacle-dense":
        from simulation.terrain.hf_env import dense_obstacle_terrain
        env_cfg.scene.terrain = dense_obstacle_terrain
    elif cfg.env_name == "omni":
        # todo: remain bugs to fix
        from simulation.terrain.omni_env import omni_terrain_cfg
        env_cfg.scene.terrain = omni_terrain_cfg(cfg.scene_id)
    elif cfg.env_name == "matterport3d":
        from simulation.terrain.mp3d_env import mp3d_terrain_cfg
        env_cfg.scene.terrain = mp3d_terrain_cfg(cfg.scene_id)
    elif cfg.env_name == "carla":
        from simulation.terrain.carla_env import carla_terrain_cfg
        env_cfg.scene.terrain = carla_terrain_cfg()
    # elif cfg.env_name == "mars":  # todo: bugs to fix
    #     from simulation.terrain.mars_terrains import mars_terrains_cfg
    #     env_cfg.scene.terrain = mars_terrains_cfg()
    else:
        raise NotImplementedError(f'[{cfg.env_name}] env has not been implemented yet')

    # ===============================================================================================
    # Environment construct
    env = ManagerBasedRLEnv(env_cfg)
    print("env.observation_manager.group_obs_term_dim", env.observation_manager.group_obs_term_dim)
    print("env.observation_manager.active_terms", env.observation_manager.active_terms['policy'])

    # ===============================================================================================
    # locomotion policy setup
    if cfg.policy == "wmp_loco":
        from locomotion.env_cfg.wmp_env import WMPObsEnvWrapper
        env = WMPObsEnvWrapper(env)
        from locomotion.policy.wmp_loco import load_policy_wmp, world_model_data_init, update_wm
        policy = load_policy_wmp(robot_name=cfg.robot_name, device=cfg.policy_device)
        depths = sm.add_camera_wmp(cfg.camera_freq)
    elif cfg.policy == "him_loco":
        from locomotion.env_cfg.him_env import HIMLocoEnvWrapper
        env = HIMLocoEnvWrapper(env)
        from locomotion.policy.him_loco import load_policy_him
        policy = load_policy_him(robot_name=cfg.robot_name, device=cfg.policy_device)
    elif cfg.policy == "legged_loco":
        from locomotion.env_cfg.legged_env import LeggedLocoEnvWrapper
        env = LeggedLocoEnvWrapper(env)
        from locomotion.policy.legged_loco import load_policy_legged
        policy = load_policy_legged(robot_name=cfg.robot_name, device=cfg.policy_device)
    elif cfg.policy == "wtw_loco":
        from locomotion.env_cfg.wtw_env import WTWLocoEnvWrapper
        # env_cfg.scene.legged_robot.init_state.joint_pos['.*L_hip_joint'] = 0.0
        # env_cfg.scene.legged_robot.init_state.joint_pos['.*R_hip_joint'] = 0.0
        # env_cfg.scene.legged_robot.init_state.joint_pos['R[L,R]_thigh_joint'] = 0.8
        # env_cfg.observations.policy.base_lin_vel.scale = 2.0
        # env_cfg.actions.joint_pos.scale = 0.25
        env = WTWLocoEnvWrapper(env)
        from locomotion.policy.wtw_loco import load_policy_wtw
        policy = load_policy_wtw(robot_name=cfg.robot_name, device=cfg.policy_device)
    else:
        raise NotImplementedError(f'Policy {cfg.policy} not implemented')

    # ===============================================================================================
    # Sensor setup
    agent_sensors.create_view_camera(cfg.view_pos, cfg.view_rot)
    cameras, lidar_annotators = None, None
    if cfg.enable_camera:
        cameras = sm.add_camera(cfg)
    if cfg.enable_lidar:
        lidar_annotators = sm.add_rtx_lidar()

    # ===============================================================================================
    # COTCalculator setup
    # from simulation.agent.util import COTCalculator
    # robot = env.unwrapped.scene.envs[0].robot
    # cot_calculator = COTCalculator(robot.data, env_cfg.sim.dt)

    # ===============================================================================================
    # local planner setup
    local_planner = None
    if cfg.test_level > 0:
        if cfg.local_planner == 'depth_planner':
            from local_planner.policy.depth_policy import LocalPlannerDepth
            local_planner = LocalPlannerDepth(cfg)
            assert cameras is not None, 'Camera not enabled'
            local_planner.set_depth_cameras(cameras)
        elif cfg.local_planner == 'rl_roverlab':
            from local_planner.policy.roverlab_rl import LocalPlannerRLRoverLab
            local_planner = LocalPlannerRLRoverLab(cfg)
        else:
            raise NotImplementedError(f'Local planner {cfg.local_planner} not implemented')

        from local_planner.env_wrapper.planner_env_wrapper import LocalPlannerEnvWrapper
        env = LocalPlannerEnvWrapper(env)
        env.set_local_planner(local_planner)

    # ===============================================================================================
    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    system_input.subscribe_to_keyboard_events(keyboard, agent_ctrl.sub_keyboard_event)
    # reset control
    reset_ctrl = agent_ctrl.AgentResetControl()
    system_input.subscribe_to_keyboard_events(keyboard, reset_ctrl.sub_keyboard_event)

    # ===============================================================================================
    # ROS2 Bridge
    if cfg.data_access_method == "ros2":
        rclpy.init()
        dm = RobotDataManager(env, lidar_annotators, cameras, cfg)

    # ===============================================================================================
    # simulation loop
    sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    obs, _ = env.reset()
    # obs_list = obs.cpu().numpy().tolist()[0]
    # obs_list = ["{:.3f}".format(v) for v in obs_list]
    print(f'[init obs shape]: {obs.shape}')

    if cfg.policy == "wmp_loco":
        # init world_model data
        world_model_data_init(obs)
    if local_planner:
        local_planner.start()

    print(f'start simulation loop: test_level {cfg.test_level}, loco_policy: {cfg.policy}, local_planner: {cfg.local_planner}')
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
            if cfg.data_access_method == "ros2":
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

    if local_planner:
        local_planner.stop()
    if cfg.data_access_method == "ros2":
        dm.destroy_node()
        rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    run_simulator()
