from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path

from .common import add_semantic_label


def create_omni_env(scene_id):
    add_semantic_label()
    assets_root_path = get_assets_root_path()
    # prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim(f"/World/{scene_id}", "Xform")
    asset_path = assets_root_path+f"/Isaac/Environments/{scene_id}.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_env():
    add_semantic_label()
    assets_root_path = get_assets_root_path()
    # prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_forklifts_env():
    add_semantic_label()
    assets_root_path = get_assets_root_path()
    # prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_shelves_env():
    add_semantic_label()
    assets_root_path = get_assets_root_path()
    # prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    prim.GetReferences().AddReference(asset_path)

def create_full_warehouse_env():
    # add_semantic_label()
    assets_root_path = get_assets_root_path()
    print(f'assets_root_path {assets_root_path}')
    # prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    prim.GetReferences().AddReference(asset_path)

def create_hospital_env():
    add_semantic_label()
    assets_root_path = get_assets_root_path()
    # prim = get_prim_at_path("/World/Hospital")
    prim = define_prim("/World/Hospital", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Hospital/hospital.usd"
    prim.GetReferences().AddReference(asset_path)

def create_office_env():
    add_semantic_label()
    assets_root_path = get_assets_root_path()
    print(f'assets_root_path {assets_root_path}')
    prim = get_prim_at_path("/World/Office")
    print(f'prim {prim}')
    prim = define_prim("/World/Office", "Xform")
    print(f'prim {prim}')
    asset_path = assets_root_path+"/Isaac/Environments/Office/office.usd"
    prim.GetReferences().AddReference(asset_path)

