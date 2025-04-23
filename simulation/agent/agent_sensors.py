import omni
import numpy as np
from pxr import Gf
import omni.replicator.core as rep
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils


class SensorManager:
    def __init__(self, num_envs, robot_name):
        self.num_envs = num_envs
        self.robot_name = robot_name
        if self.robot_name in ['A1', 'Aliengo']:
            self.base_name = 'trunk'
        else:
            self.base_name = 'base'

    def add_rtx_lidar(self):
        lidar_annotators = []
        for env_idx in range(self.num_envs):
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/lidar",
                parent=f"/World/envs/env_{env_idx}/{self.robot_name}/{self.base_name}",
                config="Hesai_XT32_SD10",
                # config="Velodyne_VLS128",
                translation=(0.2, 0, 0.2),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
            )

            annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            annotator.attach(hydra_texture.path)
            lidar_annotators.append(annotator)
        return lidar_annotators

    def add_camera(self, freq):
        cameras = []
        for env_idx in range(self.num_envs):
            camera = Camera(
                prim_path=f"/World/envs/env_{env_idx}/{self.robot_name}/{self.base_name}/front_cam",
                translation=np.array([0.27, 0.0, 0.2]),
                frequency=freq,
                resolution=(640, 480),
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
            )
            camera.initialize()
            camera.set_focal_length(1.5)
            cameras.append(camera)
        return cameras

    def add_camera_wmp(self, freq):
        resolution_size = 64
        horizontal_fov = 58.0  # 58 degrees
        # focal_length_pixel = (resolution_size / 2) / np.tan(np.radians(horizontal_fov) / 2)
        cameras = []
        for env_idx in range(self.num_envs):
            camera = Camera(
                prim_path=f"/World/envs/env_{env_idx}/{self.robot_name}/{self.base_name}/front_cam",
                translation=np.array([0.27, 0.0, 0.03]),
                frequency=freq,
                resolution=(resolution_size, resolution_size),
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
            )
            camera.initialize()
            focal_length_sim = camera.get_horizontal_aperture() / 2 / np.tan(np.radians(horizontal_fov) / 2)
            camera.set_focal_length(focal_length_sim)
            print(f'camera fov {np.degrees(camera.get_horizontal_fov())} degrees')
            camera.set_clipping_range(near_distance=0.05, far_distance=4.0)
            camera.add_distance_to_image_plane_to_frame()
            cameras.append(camera)
        return cameras

def create_view_camera(pos, rot):
    from omni.isaac.core.utils.prims import create_prim, set_prim_property
    create_prim("/World/CustomCamera", "Camera",
                translation=np.array(pos),
                orientation=rot_utils.euler_angles_to_quats(np.array(rot), degrees=True))
    # 设置焦距
    set_prim_property(
        prim_path="/World/CustomCamera",
        property_name="focalLength",
        property_value=30.0
    )