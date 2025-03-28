
from pathlib import Path
import os
import cv2
import numpy as np
import sys
import tqdm
import shutil

# script to render images from a mesh
# conda create --name cam_test python=3.9.19
# conda activate cam_test
# pip install numpy==1.26.4
# pip install open3d==0.18.0
# pip install opencv-python==4.10.0.84

# https://www.open3d.org/docs/0.17.0/tutorial/visualization/cpu_rendering.html
os.environ["EGL_PLATFORM"] = "surfaceless"  # Ubuntu 20.04+
# os.environ["OPEN3D_CPU_RENDERING"] = "true"
import open3d as o3d
import open3d.visualization.rendering as rendering

sys.path.append("")

from maxi_marvin.convert_marvin_pointclouds import load_marvin_calibration


# Load camera parameters for reprojection
folder = Path(r"camera_params_marvin")
obj = load_marvin_calibration(folder)
# obj.create_new_poses()



def create_render_obj(mesh, width=1080, height=1920):
    """Scripts that loads in a mesh, and returns a render objects"""
    render = rendering.OffscreenRenderer(width, height)
    render.scene.view.set_post_processing(False)

    mat_mesh = o3d.visualization.rendering.MaterialRecord()
    mat_mesh.shader = "defaultUnlit"
    mat_mesh.transmission = 0

    render.scene.add_geometry("mesh", mesh, mat_mesh, True)
    # render.get_render_option().background_color = [1, 1, 1]
    return render

def render_single_cam(render, cam_num, width=1080, height=1920):
    fx, cx, fy, cy = obj.get_intrinsics_fxcx_fycy(str(cam_num))

    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # https://www.open3d.org/docs/latest/python_api/open3d.visualization.rendering.OffscreenRenderer.html#open3d.visualization.rendering.OffscreenRenderer
    render.setup_camera(intrinsics, obj.get_o3d_tf(str(cam_num)))

    bgr_rendered = np.asarray(render.render_to_image())[:, :, ::-1]
    cv2.imwrite(f"rendered_{cam_num}.png", bgr_rendered)

if __name__=="__main__":
    mesh_name = "Harvest_01_PotNr_55_mesh.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_name))
    render = create_render_obj(mesh)

    render_single_cam(render, 1)
