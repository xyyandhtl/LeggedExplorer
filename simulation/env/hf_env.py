from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import FlatPatchSamplingCfg

# from training.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter
from .hf_terrain import HfUniformDiscreteObstaclesTerrainCfg, HfTunnelTerrainCfg
from .common import add_semantic_label

tunnel_terrain = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/Scene",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(3, 3),
        border_width=3,
        # border_height=3.0,
        num_rows=10,
        num_cols=1,
        color_scheme="height",
        sub_terrains={"t1": HfTunnelTerrainCfg(
            # seed=0,
            # size=(50, 50),
            obstacle_width_range=(0.5, 1.0),
            obstacle_height_range=(0.1, 1.0),
            num_obstacles=3,
            obstacles_distance=0.4,
            # border_width=5,
            platform_width=0.0,
            avoid_positions=[],
            wall_height=1.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
            }
        )},
    ),
    visual_material=None,
)

sparse_obstacle_terrain = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(50, 50),
        color_scheme="height",
        sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
            seed=0,
            size=(50, 50),
            obstacle_width_range=(0.5, 1.0),
            obstacle_height_range=(1.0, 2.0),
            num_obstacles=100 ,
            obstacles_distance=2.0,
            border_width=5,
            avoid_positions=[[0, 0]],
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
            }
        )},
    ),
    visual_material=None,
)

medium_obstacle_terrain = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(50, 50),
        color_scheme="height",
        sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
            seed=0,
            size=(50, 50),
            obstacle_width_range=(0.5, 1.0),
            obstacle_height_range=(1.0, 2.0),
            num_obstacles=200 ,
            obstacles_distance=2.0,
            border_width=5,
            avoid_positions=[[0, 0]],
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
            }
        )},
    ),
    visual_material=None,
)

dense_obstacle_terrain = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(50, 50),
        color_scheme="height",
        sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
            seed=0,
            size=(50, 50),
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.1, 1.0),
            num_obstacles=200,
            obstacles_distance=0.4,
            border_width=5,
            platform_width=2.0,
            avoid_positions=[[0, 0]],
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
            }
        )},
    ),
    visual_material=None,
)

def create_obstacle_sparse_env():
    TerrainImporter(sparse_obstacle_terrain)

def create_obstacle_medium_env():
    TerrainImporter(medium_obstacle_terrain)

def create_obstacle_dense_env():
    TerrainImporter(dense_obstacle_terrain)

def create_tunnel_env():
    TerrainImporter(tunnel_terrain)

