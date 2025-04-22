import os

from pxr import Gf
from omni.isaac.core.utils.prims import define_prim, set_prim_property


def create_carla_env():
    # 定义根原语
    prim = define_prim("/World/Carla", "Xform")
    print(f'prim {prim}')
    # 加载你自己的 USD 文件（例如 Office 环境）
    asset_path = f"{os.getenv('USER_PATH_TO_USD')}/carla_export/carla.usd"
    # 引用 USD 文件
    prim.GetReferences().AddReference(asset_path)

    set_prim_property(
        prim_path="/World/Carla",
        property_name="xformOp:translate",
        property_value=Gf.Vec3d(-200.0, -125.0, 0.0)
    )

