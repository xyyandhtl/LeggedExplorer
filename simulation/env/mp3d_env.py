import os

from omni.isaac.core.utils.prims import define_prim, set_prim_property

# todo: more flexable settings, add collider, physical material, semantic and etc when loading
def create_matterport3d_env(scene_id):
    # scene_id = "GdvgFV5R1Z5"
    # 定义根原语
    prim = define_prim("/World/Mp3d", "Xform")
    # 加载你自己的 USD 文件（例如 Office 环境）
    asset_path = f"{os.getenv('USER_PATH_TO_USD')}/mp3d_export/{scene_id}/matterport.usd"
    print(f'Added prim {prim} from asset {asset_path}')
    # 引用 USD 文件
    prim.GetReferences().AddReference(asset_path)

