import os

from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg

# todo: more flexable settings, add collider, physical material, semantic and etc when loading
def mp3d_terrain_cfg(scene_id):
    mp3d_terrain = AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=f"{os.getenv('USER_PATH_TO_USD')}/terrain/mp3d_export/{scene_id}/matterport.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    return mp3d_terrain

