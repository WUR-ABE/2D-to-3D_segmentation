import numpy as np
import pandas as pd
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def array_to_pointcloud(xyz_array, bgr_img=None):
    """Convert an array of x,y,z, coordinated and RGB colors to a pointcloud"""
    pcd_object = o3d.geometry.PointCloud()
    pcd_object.points = o3d.utility.Vector3dVector(xyz_array)
    if bgr_img is not None:
        rgb_img = bgr_img[:, :, ::-1]
        pcd_object.colors = o3d.utility.Vector3dVector(np.reshape(rgb_img, (rgb_img.shape[0] * rgb_img.shape[1], 3)) / 255.0)
    return pcd_object


def show_voxel_grid(pcd):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])


def show_octree(octree):
    o3d.visualization.draw_geometries([octree])


def image2classes(bgr_segmented_img, config_dict):
    """ "Coverts the colors in the RGB colored output image of the DL network back into classes for easy processing
    returns a binary image with the classlabel als 'color'"""
    binary_img = np.zeros(bgr_segmented_img.shape[:2], dtype=np.uint8)
    for key in config_dict["rgb_encoding"].keys():
        binary_img[np.all(bgr_segmented_img == config_dict["rgb_encoding"][key]["rgb_encoding"][::-1], axis=2)] = config_dict[
            "rgb_encoding"
        ][key]["class_id"]
    # remove pot class
    # binary_img[binary_img==config_dict['rgb_encoding']["Pot"]["class_id"]]=0
    return binary_img


def pc_class2img(df, column_name = "class", config_dict = {}):
    for key in config_dict["rgb_encoding"].keys():
        df.loc[df[column_name]==config_dict["rgb_encoding"][key]["class_id"], ["blue", "green", "red"]] = config_dict["rgb_encoding"][key]["rgb_encoding"][::-1]
    return df

def classes2img(binary_img, config_dict):
    """ "Coverts the colors in the RGB colored output image of the DL network back into classes for easy processing
    returns a binary image with the classlabel als 'color'"""
    bgr_segmented_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    for key in config_dict["rgb_encoding"].keys():
        bgr_segmented_img[binary_img == config_dict["rgb_encoding"][key]["class_id"]] = config_dict["rgb_encoding"][key][
            "rgb_encoding"
        ][::-1]
        # np.all(bgr_segmented_img == config_dict['rgb_encoding'][key]['rgb_encoding'][::-1], axis=2)
    # remove pot class
    # binary_img[binary_img==config_dict['rgb_encoding']["Pot"]["class_id"]]=0
    return bgr_segmented_img


def save_df_pointcloud(filename, df, pcd_object=None):
    """***Open3D cannot save/load pointclouds with 3D scalars fields, therefore we use plyfile ***"""
    if pcd_object is not None:
        # pcd object contains the points with colors, the dataframe the other scalar fields
        xyz = np.asarray(pcd_object.points)
        rgb = np.round(np.abs(np.asarray(pcd_object.colors)) * 255 * 255)
        rgb = rgb.astype(int)
        df_tuple = df.apply(tuple, axis=1).tolist()
        vertices = list(tuple(sub) + tuple(rgb[idx]) + df_tuple[idx] for idx, sub in enumerate(xyz.tolist()))

        # Create dictionary with datatypes
        dtype_list = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        for colname in df.columns:
            dtype_list.append((colname, "f4"))
        vertices = np.array(vertices, dtype=dtype_list)
    else:
        # The dataframe contains the points, colors and scalar fields.
        dtype_dict = {"x": "f4", "y": "f4", "z": "f4", "red": "u1", "green": "u1", "blue": "u1"}
        for colname in df.columns:
            if colname not in dtype_dict:
                dtype_dict[colname] = "f4"
        vertices = df.to_records(index=False, column_dtypes=dtype_dict)
    ply = PlyData([PlyElement.describe(vertices, "vertex")], text=False)
    ply.write(filename)


def load_df_pointcloud(filename, return_pointcloud=True, df_drop=["x", "y", "z", "red", "green", "blue"]):
    """Load pointcloud from disk, optional returns pointcloud in the dataframe or as pointcloud"""
    plydata = PlyData.read(filename)
    df = pd.DataFrame(np.array(plydata["vertex"].data))

    if return_pointcloud is True:
        # Using a view to convert an array to a recarray:
        df, pcd = df_to_pointcloud(df, df_drop)
    else:
        pcd = None
    return (df, pcd)


def df_to_pointcloud(df, df_drop=["x", "y", "z", "red", "green", "blue"]):
    """ "Conerts a dataframe to a pointcoud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(df[["x", "y", "z"]].values.tolist()))
    pcd.colors = o3d.utility.Vector3dVector(np.array(df[["red", "green", "blue"]].values.tolist()) / 255)
    df.drop(df_drop, axis=1, inplace=True)
    return (df, pcd)


def q_mult(q1, q2):
    # multipling two quaternions with each other
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    quat = [x, y, z, w]
    return quat


def inverse(quat, trans):
    rot_GT_R = R.from_quat(quat)
    rot_matrix_inv = rot_GT_R.inv().as_matrix()
    trans_inv = -1 * np.dot(rot_matrix_inv, np.array(trans).transpose())
    return R.from_matrix(rot_matrix_inv).as_quat(), trans_inv


def transform_pc_matrix(xyz_rgb, matrix, xyz_only=True):
    trans = matrix[:3, 3]
    rot_mat = matrix[:3, :3]

    if xyz_only:  # assumes that the input is an np_array of pointsx3
        return np.add(np.dot(rot_mat, xyz_rgb.transpose()).transpose(), trans)  # output is pointsx 3 matrix
    else:
        xyz_rgb_array = np.array(xyz_rgb)
        xyz_transformed = np.add(
            np.dot(rot_mat, xyz_rgb_array[:, 0:3].transpose()).transpose(), trans
        )  # output is pointsx 3 matrix
        xyz_rgb_array[:, 0:3] = xyz_transformed
        return xyz_rgb_array


def transform_pc_quat(xyz_rgb, quat, trans, xyz_only=True):
    rot_mat = R.from_quat(quat).as_matrix()

    if xyz_only:  # assumes that the input is an np_array of pointsx3
        return np.add(np.dot(rot_mat, xyz_rgb.transpose()).transpose(), trans)  # output is pointsx 3 matrix
    else:
        xyz_rgb_array = np.array(xyz_rgb)
        xyz_transformed = np.add(
            np.dot(rot_mat, xyz_rgb_array[:, 0:3].transpose()).transpose(), trans
        )  # output is pointsx 3 matrix
        xyz_rgb_array[:, 0:3] = xyz_transformed
        return xyz_rgb_array


def visualize_coordinate_system(rotation_matrix, size=0.1, save_name=None, gray=False):
    """visualize coordinates system, size is in meters, so with
    suize=0.1 x_axis is 10 """

    x_bgr = [0, 0, 255]
    y_bgr = [0, 255, 0]
    z_bgr = [255, 0, 0]

    if gray:
        x_bgr = [127, 127, 127]
        y_bgr = [127, 127, 127]
        z_bgr = [127, 127, 127]

    x_axis = rotation_matrix[:3, 0]
    y_axis = rotation_matrix[:3, 1]
    z_axis = rotation_matrix[:3, 2]

    if len(rotation_matrix[:, 0]) == 4:
        xyz = rotation_matrix[:3, 3]
    else:
        xyz = np.array([0, 0, 0])

    # print(matrix)
    x_points = create_points(x_axis, size, xyz)
    y_points = create_points(y_axis, size, xyz)
    z_points = create_points(z_axis, size, xyz)

    csv_matrix = np.zeros((300, 6), dtype=np.float32)
    csv_matrix[:100, :3] = x_points
    csv_matrix[:100, 3:] = x_bgr

    csv_matrix[100:200, :3] = y_points
    csv_matrix[100:200, 3:] = y_bgr

    csv_matrix[200:300, :3] = z_points
    csv_matrix[200:300, 3:] = z_bgr
    
    # for visualisatoin of extra cams in gray(paper)
    # csv_matrix[:, :3] = csv_matrix[:, :3]/1000
    # csv_matrix[:, 3:] = [127, 127, 127]


    df = pd.DataFrame(csv_matrix, columns=["X", "Y", "Z", "Blue", "Green", "Red"])
    if save_name is not None:
        df.to_csv(str(save_name), index=False)
    return df


def create_points(axis, size, xyz):
    axis = axis / np.linalg.norm(axis)
    return np.linspace([0, 0, 0], [axis], 100).squeeze(1) * size + xyz