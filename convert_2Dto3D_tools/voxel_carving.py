# voxel_carving.py
import cv2
import open3d as o3d

from pathlib import Path
import numpy as np
import pandas as pd

from convert_marvin_pointclouds import load_marvin_calibration
# from convert_2Dto3D_tools import pointcloud_utils
"""The maxi marvin code is not publically available, but the following code snippet is a open3d based voxel carving implementation.
Note that this implementation is not optimized for speed. 
"""

def custom_voxel_carving(obj, folder, img_list=None, imge_seg_list=None, cubic_size = [400, 400, 700], voxel_size=5):
    # set voxel grid size and resolution. Increase voxel_size to speed up.
    
    # voxel_size = 5

    print("Creating VoxelGrid with size %.2f, wait a few minutes..." % voxel_size)
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size[0],
        height=cubic_size[1],
        depth=cubic_size[2],
        voxel_size=voxel_size,
        # origin=np.array([-cubic_size[0] / 2.0, -cubic_size[1] / 2.0, -cubic_size[2] / 2.0]),
        origin=np.array([0, 0, 0], dtype=float),
        color=np.array([0, 0, 0], dtype=float))

    for cam_name in sorted(folder.glob("*preseg*.png")):
        if "coloured" in cam_name.stem:
            continue
        bgr_img = cv2.imread(str(cam_name))
        mask = np.ones((1920, 1080), dtype=np.float32)*1 # must be float otherwise does not work...
        mask[np.all(bgr_img==[0,0,0], axis=2)] = 0

        cam_num = str(int(cam_name.stem.split("-")[-1].replace("cam_","")))
        params = o3d.camera.PinholeCameraParameters()
        fx, cx, fy, cy = obj.get_intrinsics_fxcx_fycy(cam_num)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(1080, 1920, fx, fy, cx, cy)

        # params.intrinsic.set_intrinsics(1080, 1920, fx, fy, cx, cy)
        params.intrinsic = intrinsics
        params.extrinsic = obj.get_o3d_tf(cam_num)
        # True is actually better, but False matches MaxiMarvin
        voxel_carving.carve_silhouette(o3d.geometry.Image(mask), params, keep_voxels_outside_image=True) 

        print(voxel_carving)

    # create voxel points 
    xyz_array = np.asarray([voxel_carving.origin + pt.grid_index*voxel_carving.voxel_size for pt in voxel_carving.get_voxels()])
    df = pd.DataFrame(xyz_array, columns=["x", "y", "z"])/1000

    # to be compatible with marvin output we need to reproject as well  
    # for cam_num in range(0, 15):
    #     cam_num = str(cam_num)
    #     points = reproject_points(df[["x", "y", "z"]].values.astype(float), cam_num)
    #     df["x" + cam_num] = points[:, 0]
    #     df["y" + cam_num] = points[:, 1]

    # pointcloud_utils.save_df_pointcloud("keep_voxels_outside_image.ply", df)
    
    file_name = "custom_voxel_carving.csv"
    df.to_csv(file_name, index=False)
    print("saved point cloud as", file_name)

if __name__=="__main__":
    obj = load_marvin_calibrations(Path(r"camera_params_marvin"))
    folder = Path("example_data") / "inference" / "Harvest_02_PotNr_27"
    custom_voxel_carving(obj, folder, cubic_size = [400, 400, 700])