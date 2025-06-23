import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import FlatPatchSamplingCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# from training.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter
from .hf_terrain import HfUniformDiscreteObstaclesTerrainCfg, HfTunnelTerrainCfg
# from .common import add_semantic_label


ground_terrain_large = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/Terrain/Ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(100, 10),
        border_width=3,
        # border_height=3.0,
        num_rows=1,
        num_cols=4,
        color_scheme="height",
        sub_terrains={"ground": HfWaveTerrainCfg(
            amplitude_range=(0.0, 0.0),
            num_waves=2,
        )},
    ),
    visual_material=None,
)

tunnel_terrain_large = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/Terrain/Obstacles",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(100, 10),
        border_width=3,
        # border_height=3.0,
        num_rows=1,
        num_cols=4,
        color_scheme="height",
        sub_terrains={"tunnel": HfTunnelTerrainCfg(
            # seed=0,
            # size=(50, 50),
            obstacle_width_range=(0.5, 2.0),
            obstacle_height_range=(0.05, 1.50),
            num_obstacles=100,
            obstacles_distance=0.4,
            # border_width=5,
            platform_width=3.0,
            avoid_positions=[],
            wall_height=1.0,
            ground_height=-0.01,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=100, patch_radius=0.5, max_height_diff=0.2),
                "init_pos": FlatPatchSamplingCfg(num_patches=100, patch_radius=0.5, max_height_diff=0.2),
            }
        )},
    ),
    visual_material=None,
)

ground_terrain_tiny = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/Terrain/Ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(50, 5),
        border_width=3,
        # border_height=3.0,
        num_rows=1,
        num_cols=4,
        color_scheme="height",
        sub_terrains={"ground": HfWaveTerrainCfg(
            amplitude_range=(0.0, 0.0), # todo: train with rough/wave ground
            num_waves=2,
        )},
    ),
    visual_material=None,
)

tunnel_terrain_tiny = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/Terrain/Obstacles",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(30, 5),
        border_width=3,
        # border_height=3.0,
        num_rows=1,
        num_cols=4,
        color_scheme="height",
        sub_terrains={"tunnel": HfTunnelTerrainCfg(
            # seed=0,
            # size=(50, 50),
            obstacle_width_range=(0.5, 1.5),
            obstacle_height_range=(0.05, 1.50),
            num_obstacles=1,
            obstacles_distance=0.4,
            # border_width=5,
            platform_width=2.0,
            avoid_positions=[],
            wall_height=1.0,
            ground_height=0, # -0.01  0
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=100, patch_radius=0.5, max_height_diff=0.2),
                "init_pos": FlatPatchSamplingCfg(num_patches=100, patch_radius=0.5, max_height_diff=0.2),
            }
        )},
    ),
    visual_material=None,
)

@configclass
class TunnelTerrainSceneCfg(InteractiveSceneCfg):
    # terrain = ground_terrain_tiny
    obstacles = tunnel_terrain_tiny
    # combined_terrain_prim_path = "/World/Terrain/Combined"


tunnel_terrain_single = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/Terrain/Obstacles",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(50, 5),
        border_width=3,
        # border_height=3.0,
        num_rows=1,
        num_cols=1,
        color_scheme="height",
        sub_terrains={"tunnel": HfTunnelTerrainCfg(
            # seed=0,
            # size=(50, 50),
            obstacle_width_range=(0.5, 1.5),
            obstacle_height_range=(0.05, 1.50),
            num_obstacles=5,
            obstacles_distance=0.4,
            # border_width=5,
            platform_width=0.8,
            avoid_positions=[],
            wall_height=1.0,
            ground_height=0, # -0.01  0
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=100, patch_radius=0.5, max_height_diff=0.2),
                "init_pos": FlatPatchSamplingCfg(num_patches=100, patch_radius=0.5, max_height_diff=0.2),
            }
        )},
    ),
    visual_material=None,
)


sparse_obstacle_terrain = TerrainImporterCfg(
    prim_path="/World/Terrain",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(50, 50),
        color_scheme="height",
        sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
            seed=0,
            size=(50, 50),
            obstacle_width_range=(0.5, 1.0),
            obstacle_height_range=(1.0, 1.5),
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
    prim_path="/World/Terrain",
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
    prim_path="/World/Terrain",
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


# --------- stairs ---------
stairs_terrain = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=3,
            num_cols=4,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.16, 0.20),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.1, max_height_diff=0.05),
                    },
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.16, 0.20),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.1, max_height_diff=0.05),
                    },
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.1, max_height_diff=0.05),
                    },
                ),
                "hf_pyramid_slope": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.1, max_height_diff=0.05),
                    },
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.1, max_height_diff=0.05),
                    },
                ),
                "box": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.1), platform_width=2.0,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.1, max_height_diff=0.05),
                    },
                ),
            },
        ),
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

def create_obstacle_sparse_env():
    TerrainImporter(sparse_obstacle_terrain)

def create_obstacle_medium_env():
    TerrainImporter(medium_obstacle_terrain)

def create_obstacle_dense_env():
    TerrainImporter(dense_obstacle_terrain)

def create_tunnel_env():
    TerrainImporter(tunnel_terrain_large)

