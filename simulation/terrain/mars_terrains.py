import os

from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from training.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter

base_path = f"{os.getenv('USER_PATH_TO_USD')}/terrains"
print(f'terrain base_path {base_path}')
ground_terrain_path = os.path.join(base_path, "mars", "terrain_only.usd")
obstacles_path = os.path.join(base_path, "mars", "rocks_merged.usd")
hidden_terrain_path = os.path.join(base_path, "mars", "terrain_merged.usd")


def mars_terrains_cfg():
    terrain = TerrainImporterCfg(
        class_type=RoverTerrainImporter,
        prim_path="/World/Terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=ground_terrain_path,
    )
    return terrain

@configclass
class MarsTerrainSceneCfg(InteractiveSceneCfg):
    """
    Mars Terrain Scene Configuration
    """
    # Hidden Terrain (merged terrain of ground and obstacles) for raycaster.
    # This is done because the raycaster doesn't work with multiple meshes
    hidden_terrain = AssetBaseCfg(
        prim_path="/World/terrain/hidden_terrain",
        spawn=sim_utils.UsdFileCfg(
            visible=False,
            usd_path=hidden_terrain_path,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Ground Terrain
    terrain = TerrainImporterCfg(
        class_type=RoverTerrainImporter,
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=ground_terrain_path,
    )

    # Obstacles
    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=obstacles_path,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
