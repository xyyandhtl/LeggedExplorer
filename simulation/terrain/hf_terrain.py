from isaaclab.terrains.height_field.utils import height_field_to_mesh
import numpy as np
import time

@height_field_to_mesh
def uniform_discrete_obstacles_terrain(difficulty: float, cfg) -> np.ndarray:
    np.random.seed(cfg.seed) 
    def is_good_position(obs_list, obs_pos, min_dist):
        for i in range(len(obs_list)):
            obs_pos_i = obs_list[i]
            dist = ((obs_pos[0] - obs_pos_i[0])**2 + (obs_pos[1] - obs_pos_i[1])**2)**0.5
            if (dist < min_dist):
                return False
        return True

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height_min = cfg.obstacle_height_range[0]
    obs_height_max = cfg.obstacle_height_range[1]
    obs_width_min = cfg.obstacle_width_range[0]
    obs_width_max = cfg.obstacle_width_range[1]
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    obs_dist = cfg.obstacles_distance
    stop_sampling = False
    # generate the obstacles
    obs_list = cfg.avoid_positions
    for _ in range(cfg.num_obstacles):
        # sample size
        height = int(np.random.uniform(obs_height_min, obs_height_max) / cfg.vertical_scale)
        width = int(np.random.uniform(obs_width_min, obs_width_max) / cfg.horizontal_scale)
        length = int(np.random.uniform(obs_width_min, obs_width_max) / cfg.horizontal_scale)

        # sample position
        start_time = time.time()
        good_position = False
        while (not good_position):
            x_start = int(np.random.choice(obs_x_range))
            y_start = int(np.random.choice(obs_y_range))
            x_scale = x_start * cfg.horizontal_scale
            y_scale = y_start * cfg.horizontal_scale
            good_position = is_good_position(obs_list, [x_scale, y_scale], obs_dist)
            sample_time = time.time() 
            if (sample_time - start_time) > 0.2:
                stop_sampling = True
                break
        if (stop_sampling):
            break

        obs_list.append([x_scale, y_scale])
        # clip start position to the terrain
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height


    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)

@height_field_to_mesh
def tunnel_obstacles_terrain(difficulty: float, cfg) -> np.ndarray:
    # np.random.seed(cfg.seed)
    def is_good_position(obs_list, obs_pos, min_dist):
        for i in range(len(obs_list)):
            obs_pos_i = obs_list[i]
            dist = ((obs_pos[0] - obs_pos_i[0])**2 + (obs_pos[1] - obs_pos_i[1])**2)**0.5
            if (dist < min_dist):
                return False
        return True

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height_min = cfg.obstacle_height_range[0]
    obs_height_max = cfg.obstacle_height_range[1]
    obs_width_min = cfg.obstacle_width_range[0]
    obs_width_max = cfg.obstacle_width_range[1]
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(3, length_pixels - 3, 4)

    ground_height = int(cfg.ground_height / cfg.vertical_scale)
    # create a terrain with a flat platform at the center
    hf_raw = ground_height * np.ones((width_pixels, length_pixels))
    obs_dist = cfg.obstacles_distance
    stop_sampling = False
    # generate the obstacles
    obs_list = cfg.avoid_positions
    for _ in range(cfg.num_obstacles):
        # sample size
        height = int(np.random.uniform(obs_height_min, obs_height_max) / cfg.vertical_scale)
        width = int(np.random.uniform(obs_width_min, obs_width_max) / cfg.horizontal_scale)
        length = int(np.random.uniform(obs_width_min, obs_width_max) / cfg.horizontal_scale)

        # sample position
        start_time = time.time()
        good_position = False
        while (not good_position):
            x_start = int(np.random.choice(obs_x_range))
            y_start = int(np.random.choice(obs_y_range))
            x_scale = x_start * cfg.horizontal_scale
            y_scale = y_start * cfg.horizontal_scale
            good_position = is_good_position(obs_list, [x_scale, y_scale], obs_dist)
            sample_time = time.time()
            if (sample_time - start_time) > 0.2:
                stop_sampling = True
                break
        if (stop_sampling):
            break

        obs_list.append([x_scale, y_scale])
        # clip start position to the terrain
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height

    # add side-walls
    wall_height = int(2.0 / cfg.vertical_scale)
    hf_raw[:, 0:3] = wall_height
    hf_raw[:, length_pixels - 3:] = wall_height

    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = int(cfg.ground_height / cfg.vertical_scale)
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class HfUniformDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = uniform_discrete_obstacles_terrain
    seed: float = 0
    """Env seed."""
    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    obstacles_distance: float = MISSING
    """The minimum distance between obstacles (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    avoid_positions: list[list[float, float]] = []

@configclass
class HfTunnelTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = tunnel_obstacles_terrain
    # seed: float = 0
    """Env seed."""
    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    obstacles_distance: float = MISSING
    """The minimum distance between obstacles (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    avoid_positions: list[list[float, float]] = []
    wall_height: float = 1.0
    ground_height: float = -0.01
