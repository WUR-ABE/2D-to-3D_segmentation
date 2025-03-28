import cv2

from pathlib import Path
import numpy as np

from convert_2Dto3D_tools.convert_marvin_pointclouds import load_marvin_calibration

## an example script if you want to reproject the points to the camera images using OPENCV

def reproject_points(obj, points, cam_num):
    """Function that projects the xyz points nx3 in [mm] to pixel coordinates of given camera"""
    rvec, _ = cv2.Rodrigues(obj.get_o3d_tf(cam_num)[:3,:3])
    pixels, _ = cv2.projectPoints(points, 
                            rvec = rvec, 
                            tvec = obj.get_o3d_tf(cam_num)[:3,-1], 
                            cameraMatrix=obj.get_intrinsics(cam_num),
                            distCoeffs=np.zeros(5)
                            )
    return pixels.reshape(-1, 2).astype(int) # [[x, y]] pixels
    

def main_reproject(obj, cam_nums=["15"], coordinates_df=None, unmirror=False):
    """Function that reprojects all points (nx3) of coordinates_df using to each camera.
    Coordinates projected outside camera coordinates are set to nan. Returns updated coordinates_df"""
    print("Starting reprojection extra cams...")
    new_df = coordinates_df.copy()

    if unmirror:
        new_df[["x", "y", "z"]] = coordinates_df[["x", "y", "z"]] * [-1, 1, 1] + [400, 0, 0]
    for cam_num in cam_nums:
        pixel_coord = reproject_points(obj, new_df[["x", "y", "z"]].values.astype(float), cam_num)
        pixel_coord = pixel_coord.astype(float)
        pixel_coord[np.any(pixel_coord >= [1080, 1920], axis=1), :] = np.nan

        new_df["x" + cam_num] = pixel_coord[:, 0]
        new_df["y" + cam_num] = pixel_coord[:, 1]

        # for debug
        # dummy = np.zeros((1920, 1080), dtype=np.uint8)
        # pixel_coord2 = pixel_coord[np.all([pixel_coord[:,1]<1920,pixel_coord[:,0]<1080],axis=0), :]
        # dummy[pixel_coord2[:,1], pixel_coord2[:,0]] = 255
    if unmirror:
        # of course to use the funcion in the reprojection unmirror again
        new_df[["x", "y", "z"]] = (new_df[["x", "y", "z"]] - [400, 0, 0]) / [-1, 1, 1] 
    print("Finished extra cam reprojection")

    return new_df


if __name__=="__main__":
    obj = load_marvin_calibration(Path(r"camera_params_marvin"))
    obj.create_new_poses()

    # point = np.array([[178, 140, 442]], dtype=float) # should result in 427,771
    # point2 = point + [[5,5,5]]
    # main_reproject(obj)