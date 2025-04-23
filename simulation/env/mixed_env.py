from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainImporter
from omni.isaac.lab.terrains import TerrainGeneratorCfg
from omni.isaac.lab.terrains.trimesh import *
from omni.isaac.lab.terrains.height_field import *


def create_mixed_terrain_env():
    terrain = TerrainImporterCfg(
        prim_path="/World/complexTerrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(3, 3),  # 总地图尺寸
            border_width=1.0,
            # border_height=3.0,
            num_rows=10,
            num_cols=1,
            color_scheme="height",
            sub_terrains={
                # "stairs": HfPyramidStairsTerrainCfg(proportion=0.2, step_height_range=(0.1, 0.3),
                #                                     step_width=0.3),
                # "step": HfSteppingStonesTerrainCfg(proportion=0.2, stone_distance_range=(0.3, 0.6),
                #                                    stone_height_max=0.05, stone_width_range=(0.2, 0.4), ),
                # "slope": HfPyramidSlopedTerrainCfg(proportion=0.2, slope_range=(0.2, 0.6)),
                "stairs": HfInvertedPyramidStairsTerrainCfg(proportion=0.2, step_height_range=(0.1, 0.2),
                                                            step_width=0.3),
                # "wave": HfWaveTerrainCfg(proportion=0.2, amplitude_range=(0.1, 0.3)),
                # # "slope": HfInvertedPyramidSlopedTerrainCfg(slope_range=(0.2, 0.5)),
                # "rough": HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(0.05, 0.2), noise_step=0.01,),
                # "obstacles": HfDiscreteObstaclesTerrainCfg(proportion=0.1,
                #                                            obstacle_width_range=(1.0, 2.0),
                #                                            obstacle_height_range=(0.1, 1.0), num_obstacles=50),
                # "boxes": MeshRandomGridTerrainCfg(proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.1),
                #                                   platform_width=2.0),
                "gaps": MeshGapTerrainCfg(proportion=0.2, gap_width_range=(0.3, 1.0), platform_width=1.0),
                # "pit": MeshPitTerrainCfg(proportion=0.2, pit_depth_range=(0.2, 0.5)),
                "boxes": MeshRepeatedBoxesTerrainCfg(proportion=0.2, object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=30, height=1.0, size=(0.3, 0.3), max_yx_angle=0.0, degrees=True
                ), object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=50, height=1.0, size=(0.3, 0.3), max_yx_angle=0.0, degrees=True
                ), platform_width=0.),
                "ring": MeshFloatingRingTerrainCfg(proportion=0.2, ring_width_range=(0.8, 1.0),
                                                   ring_height_range=(0.3, 0.6), ring_thickness=0.5,
                                                   platform_width=0.0),
                "rough": HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(0.02, 0.1), noise_step=0.02
                ),
                # "rails": MeshRailsTerrainCfg(
                #     proportion=0.2, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.5, 1.0)
                # )


                # "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                #     proportion=0.1,
                #     step_height_range=(0.05, 0.17),
                #     step_width=0.3,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),
                # "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
                #     proportion=0.2,
                #     step_height_range=(0.05, 0.17),
                #     step_width=0.3,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),

                # "flat": MeshPlaneTerrainCfg(proportion=0.1),
                # "random_rough": HfRandomUniformTerrainCfg(
                #     proportion=0.01, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
                # ),

                # "init_pos": HfDiscreteObstaclesTerrainCfg(
                #     proportion=0.6,
                #     num_obstacles=10,
                #     obstacle_height_mode="fixed",
                #     obstacle_height_range=(0.3, 2.0), obstacle_width_range=(0.4, 1.0),
                #     platform_width=0.0
                # ),
                # "ring": MeshFloatingRingTerrainCfg(proportion=0.1, ring_width_range=(0.3, 0.6),
                #                                    ring_height_range=(0.3, 0.6), ring_thickness=0.5,
                #                                    platform_width=0.0),

                # "cylinder": MeshRepeatedCylindersTerrainCfg(
                #     proportion=0.1,
                #     platform_width=0.0,
                #     object_type="cylinder",
                #     object_params_start=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                #         num_objects=4,
                #         height=1.0,
                #         radius=0.5
                #     ),
                #     object_params_end=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                #         num_objects=8,
                #         height=0.6,
                #         radius=0.2
                #     ),
                # )
            },
        ),
        visual_material=None,
    )
    TerrainImporter(terrain)