import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys
from scipy.optimize import leastsq

sys.path.append("")
from convert_2Dto3D_tools.pointcloud_utils import transform_pc_quat, q_mult

# , load_df_pointcloud, save_df_pointcloud
from pathlib import Path


"""The marvin points clouds are actually mirrored and x_axis multiplied with -1.
    As a result the point clouds from the marvin do not match with the original RGB images.
    This "inccorect format" is below referred as marvin_output.

    This script provides varios tools to calculate reconstruct the point cloud
    as in orginally in calibrated in Halcon. 

    Currently everything is hard programmed for a cube dimension of
    [400,400,700]
    """
# backup of halcon position and rotation
# halcon_position = np.array([[-0.288102466,-0.862750001,-0.96585299],
#                             [-1.061569286,-0.84816119,-0.012063542],
#                             [-0.388665149,-0.818301891,1.014782195],
#                             [0.79596646,-0.811013665,0.692864966],
#                             [0.857093221,-0.830449376,-0.52633915],
#                             [-0.296034332,-0.099760371,-0.963952694],
#                             [-1.047498114,-0.093377118,-0.023203298],
#                             [-0.390855524,-0.064062563,0.980066465],
#                             [0.773436853,-0.053068745,0.668225655],
#                             [0.838120553,-0.081759106,-0.529307393],
#                             [-0.309417906,0.631040173,-0.989795615],
#                             [-1.068297159,0.639067796,-0.03981156],
#                             [-0.400901899,0.676034337,0.979825634],
#                             [0.773809883,0.683887996,0.661326372],
#                             [0.837146629,0.655042808,-0.553418408]])*1000

# halcon_rotation_deg = [[321.807012,12.29108854,9.440709408],
#                     [274.1652246,54.05972075,83.27644074],
#                     [217.1400331,17.10065706,165.2953027],
#                     [227.1696979,321.7518579,213.9199223],
#                     [305.7179497,316.8008654,317.440913],
#                     [356.3875694,14.51215344,0.971289845],
#                     [317.1016014,84.75526093,41.47854142],
#                     [182.4419039,20.560103,177.9066777],
#                     [182.2887083,309.5962793,182.0322068],
#                     [354.7265236,303.659964,356.3277881],
#                     [33.21492056,13.24413574,352.711057],
#                     [84.28746517,57.92695834,274.8073653],
#                     [142.684702,18.74563936,192.5170868],
#                     [131.2665655,320.1941599,144.5251786],
#                     [49.62679405,317.3419751,40.02287616]]


def halcon_pose_to_python(file_name):
    """Script that open a halcon.dat file and reads the camera pose
    # Rotation angles [deg] or Rodriguez vector:
    r 85.1571894843122 25.9184327337733 103.143093347494

    # Translation vector (x y z [m]):
    t -0.36474602206707 -0.184127813128111 0.0164170688185939

    should give:
    trans = [-0.364,-0.184,0.016]
    quat_xyzw = [-0.539,0.413,-0.656,-0.327] or [0.539,-.413,0.656,0.327]

    return: quaterion xyzw and translation
    """

    halcon_type = None
    r = None
    t = None

    with file_name.open("r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            elif line.startswith("f"):
                halcon_type = line.split(" ")[-1]
            elif line.startswith("r"):
                r = np.array(line.split(" ")[1:]).astype(float)
            elif line.startswith("t"):
                trans = np.array(line.split(" ")[1:]).astype(float)
            # print(line)

    # halcon has multiple rotation types. At the moment only tested for one
    if halcon_type == "0":
        quat_xyzw = R.from_euler("XYZ", r, degrees=True).as_quat()

    trans_mm = trans * 1000

    return quat_xyzw, trans_mm


def halcon_matrix_to_python(file_name):
    """The txt matrixes are actually made using the following code in halcon:
    <l>name:='cam015_param'</l>
    <c></c>
    <l>read_cam_par('W:/PROJECTS/VisionRoboticsData/NPEC_deeplearning/NPEC_Calibration_27-05-2021_02/Camera_params/'+name+'.dat', CameraParam)</l>
    <l>change_radial_distortion_cam_par ('adaptive', CameraParam, 0, CamParamOut)</l>
    <l>cam_par_to_cam_mat (CamParamOut, CameraMatrix, ImageWidth, ImageHeight)</l>
    <l>write_tuple(CameraMatrix, 'W:/PROJECTS/VisionRoboticsData/NPEC_deeplearning/NPEC_Calibration_27-05-2021_02/Camera_params/'+name+'_matrix.txt')</l>

    returns
    camera_matrix 3x3 with [[fx, 0 cx], [0, fy, cy], [0, 0, 1]]

    """
    # Open the text file
    with open(file_name, "r") as file:
        # Read each line
        camera_list = []
        for line in file:
            # Split the line by space and take the second element
            number = line.split()
            if len(number) == 1:
                continue
            else:
                camera_list.append(float(number[1]))

    return np.array(camera_list).reshape(3, 3)


class load_marvin_calibration:
    def __init__(self, calib_folder) -> None:
        self.calib_folder = calib_folder
        self.cam_list = ["cam" + str(x).zfill(3) for x in range(15)]
        self.load_pose()
        self.load_camera_matrix()
        pass

    def load_pose(self):
        """Load the calibrated poses from the marvin and convert to correct marvin output"""
        self.halcon_pose_dict = {}
        self.mirrored_pose_dict = {}  # coordinate system van marvin, mirrored so technically not correct...
        self.umo_pose_dict = {}  # umo = unmirrored marvin output
        self.o3d_pose_dict = {}  # o3d = open3d format

        for cam_num in self.cam_list:
            key = str(int(cam_num.strip("cam")))
            file_name = self.calib_folder / (cam_num + "_pose.dat")

            self.halcon_pose_dict[key] = {}
            self.mirrored_pose_dict[key] = {}
            self.umo_pose_dict[key] = {}
            self.o3d_pose_dict[key] = {}

            # Loads halcon files
            quat_xyzw, trans = halcon_pose_to_python(file_name)
            self.halcon_pose_dict[key]["quat_xyzw"] = quat_xyzw
            self.halcon_pose_dict[key]["trans"] = trans

            # To correctly convert halcon_pose to unmirrored marvin pose
            _, trans_mmo = halcon_pose2mirrored_marvin_output(quat_xyzw, trans)
            # self.umo_pose_dict[key]["quat_xyzw"] = quat_xyzw_umo
            self.mirrored_pose_dict[key]["trans"] = trans_mmo

            # To correctly convert halcon_pose to unmirrored marvin pose
            quat_xyzw_umo, trans_umo = halcon_pose2unmirrored_marvin_output(quat_xyzw, trans)
            self.umo_pose_dict[key]["quat_xyzw"] = quat_xyzw_umo
            self.umo_pose_dict[key]["trans"] = trans_umo

            # To render images in open3D different coordinate system is used
            quat_xyzw_o3d, trans_o3d = unmirrored_marvin_output2open3d(None, quat_xyzw_umo, trans_umo)
            self.o3d_pose_dict[key]["quat_xyzw"] = quat_xyzw_o3d
            self.o3d_pose_dict[key]["trans"] = trans_o3d

        print("Succesfully loaded cameras poses")

    def load_camera_matrix(self):
        """Load the intrinsics of the cameras"""
        self.intrinsic_matrix = {}
        for cam_num in self.cam_list:
            key = str(int(cam_num.strip("cam")))

            file_name = self.calib_folder / (cam_num + "_param_matrix.txt")
            matrix = halcon_matrix_to_python(file_name)
            self.intrinsic_matrix[key] = matrix

        print("Succesfully loaded camera intrinsics")

    def pose_2_tf(self, input_dict):
        tf = np.eye(4)
        tf[:3, :3] = R.from_quat(input_dict["quat_xyzw"]).as_matrix()
        tf[:3, 3] = input_dict["trans"]
        return tf

    def get_halcon_pose(self, cam_num):
        return self.halcon_pose_dict[cam_num]

    def get_halcon_tf(self, cam_num):
        return self.pose_2_tf(self.halcon_pose_dict[cam_num])

    def get_mmo_trans(self, cam_num):
        return self.mirrored_pose_dict[cam_num]["trans"]

    def get_mmo_array_cams(self, cam_nums=[15]):
        """Returns mirrored marvin cam locations"""
        cam_locations = {}
        for cam_num in cam_nums:
             cam_locations[str(cam_num)] = np.array(self.get_mmo_trans(str(cam_num))).round().astype(int)
        return cam_locations

    def get_umo_pose(self, cam_num):
        return self.umo_pose_dict[cam_num]

    def get_umo_tf(self, cam_num):
        return self.pose_2_tf(self.umo_pose_dict[cam_num])

    def get_o3d_pose(self, cam_num):
        return self.o3d_pose_dict[cam_num]

    def get_o3d_tf(self, cam_num):
        return self.pose_2_tf(self.o3d_pose_dict[cam_num])

    def get_intrinsics(self, cam_num):
        return self.intrinsic_matrix[cam_num]

    def get_intrinsics_fxcx_fycy(self, cam_num):
        return (
            self.intrinsic_matrix[cam_num][0][0],
            self.intrinsic_matrix[cam_num][0][2],
            self.intrinsic_matrix[cam_num][1][1],
            self.intrinsic_matrix[cam_num][1][2],
        )

    def get_new_tf(self, x1, x2, height_z, cam_num):
        """Script that calculates average quaternion and positoin using camera x1 and x2,
        saves the result in self.umo_pose_dict()
        input
        -----
        x1: string camera number 1
        x2: string camera number 2
        height_z: float with desired camera height
        cam_num: string/int key to store new camera pose
        """
        xy = (self.get_umo_tf(str(x1))[:2, 3] + self.get_umo_tf(str(x2))[:2, 3]) / 2
        # interpolate between horizontal camera
        quat_xyzw_o3d = self.get_umo_pose(str(x1))["quat_xyzw"]
        quat_xyzw_o3d2 = self.get_umo_pose(str(x2))["quat_xyzw"]
        key_rots = R.from_quat([quat_xyzw_o3d, quat_xyzw_o3d2])
        slerp = Slerp([0, 1], key_rots)
        interp_rots = slerp([0.5])
        # interpolate between vertical camera
        quat_xyzw_o3d_5 = self.get_umo_pose(str(x1 + 5))["quat_xyzw"]
        quat_xyzw_o3d2_5 = self.get_umo_pose(str(x2 + 5))["quat_xyzw"]

        key_rots = R.from_quat([quat_xyzw_o3d_5, quat_xyzw_o3d2_5])
        slerp = Slerp([0, 1], key_rots)
        interp_rots_5 = slerp([0.5])
        # interpolate between both horizontal and vertical camera
        key_rots = R.from_quat([interp_rots.as_quat()[0], interp_rots_5.as_quat()[0]])
        slerp = Slerp([0, 1], key_rots)
        interp_rots = slerp([0.5])

        trans_o3d = np.array([xy[0], xy[1], height_z])

        self.umo_pose_dict[str(cam_num)] = {}
        self.umo_pose_dict[str(cam_num)]["quat_xyzw"] = list(interp_rots.as_quat()[0])
        self.umo_pose_dict[str(cam_num)]["trans"] = trans_o3d

        self.intrinsic_matrix[str(cam_num)] = self.intrinsic_matrix[str(x1)].copy()

    def create_new_poses(self):
        """Function that create a two new rings of cameras"""
        ring1_height = (self.get_umo_tf("0")[2, 3] + self.get_umo_tf("5")[2, 3]) / 2

        cam_num = 15
        for x in range(0, 4):
            self.get_new_tf(x, x + 1, ring1_height, cam_num)
            cam_num += 1

        self.get_new_tf(x + 1, 0, ring1_height, cam_num)
        cam_num += 1

        # second ring
        ring2_height = (self.get_umo_tf("5")[2, 3] + self.get_umo_tf("10")[2, 3]) / 2
        for x in range(5, 9):
            self.get_new_tf(x, x + 1, ring2_height, cam_num)
            cam_num += 1
        self.get_new_tf(x + 1, 5, ring2_height, cam_num)

        for key in self.umo_pose_dict.keys():
            self.o3d_pose_dict[key] = {}
            quat_xyzw_o3d, trans_o3d = unmirrored_marvin_output2open3d(
                None, self.umo_pose_dict[key]["quat_xyzw"], self.umo_pose_dict[key]["trans"]
            )
            self.o3d_pose_dict[key]["quat_xyzw"] = quat_xyzw_o3d
            self.o3d_pose_dict[key]["trans"] = trans_o3d

            # self.mirrored_pose_dict[key] = {}
            if self.mirrored_pose_dict.get(key) is None:
                self.mirrored_pose_dict[key] = {}
                self.mirrored_pose_dict[key]["trans"] = list(self.umo_pose_dict[key]["trans"] * [-1, 1, 1] + [400, 0, 0])



    def new_poses_to_txt(self):
        """Function to visualize camera poses"""
        from pointcloud_utils import visualize_coordinate_system

        for key in self.umo_pose_dict.keys():
            save_name = "cam_pos/" + key + ".txt"
            tf = self.get_umo_tf(key)
            tf[:3, 3] = tf[:3, 3] / 1000

            gray=False
            if int(key)>15:
                gray=True
            visualize_coordinate_system(rotation_matrix=tf, size=0.250, save_name=save_name, gray=gray)
        ## add circular camera positions
        self.save_circle(cam_nums=[0, 1, 2, 3, 4], name="ring_top")
        self.save_circle(cam_nums=[5, 6, 7, 8, 9], name="ring_mid")
        self.save_circle(cam_nums=[10, 11, 12, 13, 14], name="ring_bottom")
        self.save_circle(cam_nums=[15, 16, 17, 18, 19], name="virtual_top")
        self.save_circle(cam_nums=[20, 21, 22, 23, 24], name="virtual_bottom")


    def save_circle(self, cam_nums=None, name="ring1"):
        """
        Script to visualize the camera positions in a circle
        """
        cam_locations = self.get_mmo_array_cams(cam_nums)
        points_3d = np.array(list(cam_locations.values()))
        # fit circle through the points
        # 1. Fit a plane to the 3D points
        normal, centroid = fit_plane(points_3d)

        # 2. Project the 3D points onto the best-fitting plane
        points_2d, u, v, centroid = project_points_to_plane(points_3d, normal, centroid)

        # 3. Fit a circle to the 2D points on the plane
        center_2d, radius = fit_circle_2d(points_2d)

        # 4. Reconstruct the circle in 3D
        circle_3d = reconstruct_circle_3d(center_2d, radius, u, v, centroid)
        df = pd.DataFrame(circle_3d, columns=["x", "y", "z"]) / 1000
        df[["blue", "green", "red"]] = [0, 0, 0]
        df.to_csv(f"cam_pos/{name}.csv", index=False)

    def export_to_json(self):
        for x in self.cam_list:
            key = str(int(x.strip("cam")))

            temp_dict = {}
            temp_dict["info"] = f"Exported camera parameters in open3D format of camera number: {key}. Extrinsics is transformation matrix, with translation in meters"
            fx, cx, fy, cy = self.get_intrinsics_fxcx_fycy(cam_num=key)
            temp_dict["intrinsics"] = {
                "height": 1920,
                "width": 1080,
                "fx": fx,
                "cx": cx,
                "fy": fy,
                "cy": cy
            }

            tf = self.get_o3d_tf(cam_num=key)
            tf[:3,3] = tf[:3,3]/1000
            tf = tf.tolist()


            temp_dict["extrinsics"] = tf
            import json
            with open(str(key)+".json", "w") as f:
                json.dump(temp_dict, f, indent=4)





def marvin_output2halcon_output(df, rgb=False):
    """Converts the output marvin, to the halcon format"""
    df = marvin_output2unmirrored_marvin_output(df)

    quat_inv = [0.5, 0.5, -0.5, 0.5]
    trans_inv = [-200, 350, 200]

    if rgb:
        new_xyz = transform_pc_quat(df[["x", "y", "z", "red", "green", "blue"]].values, quat_inv, trans_inv, xyz_only=False)
        df_halcon_format = pd.DataFrame(new_xyz, columns=["x", "y", "z", "red", "green", "blue"])
    else:
        new_xyz = transform_pc_quat(df[["x", "y", "z"]].values, quat_inv, trans_inv)
        df_halcon_format = pd.DataFrame(new_xyz, columns=["x", "y", "z"])

    return df_halcon_format


def marvin_output2unmirrored_marvin_output(df):
    """to convert the point cloud"""
    df["x"] = df.x * -1 + 400
    return df


def halcon_pose2mirrored_marvin_output(quat_halcon, trans_halcon):
    """Converts the halcon calibrated poses to mirrored coordinate system
    (NOT in agreement with RGB images), but this is the coordinate system from the marvin..."""
    # this shows how the camera position are incorrectly! tranformed
    # we still show this, because it is the only method to debug the generated .csv by the marvin
    # cube_dimension_xyz = [400, 400, 700]
    # marvin_position = [[pos[2]+cube_dimension_xyz[0]/2,
    #                     pos[0]+cube_dimension_xyz[1]/2,
    #                     -pos[1]+cube_dimension_xyz[2]/2] for pos in halcon_position]
    cube_dimension_xyz = [400, 400, 700]

    trans_mmo = [
        trans_halcon[2] + cube_dimension_xyz[0] / 2,
        trans_halcon[0] + cube_dimension_xyz[1] / 2,
        -trans_halcon[1] + cube_dimension_xyz[2] / 2,
    ]
    return None, trans_mmo


# def halcon_pose2unmirrored_marvin_output(cam_num):
def halcon_pose2unmirrored_marvin_output(quat_halcon, trans_halcon):
    """Converts the halcon calibrated poses to non mirrored coordinate system
    (in agreement with rgb images)"""
    # quat_halcon = R.from_euler("XYZ", halcon_rotation_deg[cam_num], degrees=True).as_quat()

    # hard code conversion from halcon to marvin
    quat = [-0.5, -0.5, 0.5, 0.5]  # x=-90, y=-90, z=0
    trans = [200, 200, 350]

    quat_xyzw_umo = q_mult(quat, quat_halcon)
    # q_new = q_mult(quat, quat_halcon)
    # matrix = R.from_quat(q_new).as_matrix()
    # tf = np.eye(4, dtype=float)
    # tf[:3,:3] = matrix
    # tf[:3,3] = transform_pc_quat(np.array(halcon_position[cam_num]), quat, trans=trans)
    trans_umo = transform_pc_quat(trans_halcon, quat, trans=trans)

    return quat_xyzw_umo, trans_umo


def z_up2y_up(df):
    quat = R.from_euler("XYZ", [90, 0, 0], degrees=True).as_quat()
    df_yup = transform_pc_quat(df[["x", "y", "z"]].values, quat, [0, 0, 0])


def unmirrored_marvin_output2open3d(tf=None, quat=None, trans=None):
    """Also works for open3d2unmirrored_marvin_output"""
    # copied from camtools R_t_to_C
    if tf is not None:
        r, t = tf[:3, :3], tf[:3, 3]
        new_trans = -r.T @ t
        new_tf = np.eye(4)
        new_tf[:3, :3] = r.T
        new_tf[:3, 3] = new_trans
        return new_tf
    elif quat is not None and trans is not None:
        r = R.from_quat(quat).as_matrix()
        new_trans = -r.T @ trans
        new_quat = R.from_quat(quat).inv().as_quat()
        return new_quat, new_trans

    else:
        print("tf or quat/trans not specified in unmirored_marvin_output2open3d...")


def get_fov(fx, fy, width=1080, height=1920):
    fov_x = np.rad2deg(2 * np.arctan2(width, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(height, 2 * fy))
    print(f"fov_x: {fov_x}, fov_y: {fov_y}")



def fit_plane(points):
    """Fit a plane to a set of 3D points using SVD."""
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]  # The normal vector of the plane
    return normal, centroid

def project_points_to_plane(points, normal, centroid):
    """Project 3D points onto the best-fitting plane."""
    projected_points = points - ((points - centroid) @ normal[:, None]) * normal
    # Create a local coordinate system (u, v) on the plane
    u = np.cross([0, 0, 1], normal)
    if np.linalg.norm(u) < 1e-6:  # Handle edge case where normal is [0, 0, 1]
        u = np.array([1, 0, 0], dtype='float64')
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    # Convert 3D points to 2D (u, v) coordinates
    local_2d_coords = np.array([(p - centroid) @ np.array([u, v]).T for p in projected_points])
    return local_2d_coords, u, v, centroid

def fit_circle_2d(points_2d):
    """Fit a circle to a set of 2D points using least-squares minimization."""
    def calc_R(xc, yc):
        """Calculate distance of each point from center (xc, yc)."""
        return np.sqrt((points_2d[:, 0] - xc)**2 + (points_2d[:, 1] - yc)**2)
    
    def cost(params):
        """Cost function to minimize: difference between distances and mean radius."""
        xc, yc = params
        R = calc_R(xc, yc)
        return R - np.mean(R)
    
    x0 = np.mean(points_2d, axis=0)  # Initial guess for center
    center, _ = leastsq(cost, x0)
    R = np.mean(calc_R(*center))  # Calculate radius
    return center, R

def reconstruct_circle_3d(center_2d, radius, u, v, centroid, n_points=100):
    """Reconstruct the 3D coordinates of the circle using its 2D center, radius, and plane orientation."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle_2d = np.array([center_2d[0] + radius * np.cos(theta), 
                          center_2d[1] + radius * np.sin(theta)]).T
    circle_3d = np.array([centroid + u * x + v * y for x, y in circle_2d])
    return circle_3d

if __name__ == "__main__":
    folder = Path(r"camera_params_marvin")
    obj = load_marvin_calibration(folder)
    obj.create_new_poses()
    obj.new_poses_to_txt()

    fx, _, fy, _ = obj.get_intrinsics_fxcx_fycy("0")
    print(get_fov(fx, fy, 1080, 1920))
    obj.export_to_json()
    # print(obj.get_o3d_tf(1))
