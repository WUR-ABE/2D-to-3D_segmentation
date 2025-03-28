"""Reprojection of the segmented rgb images on the pointcloud """
import datatable as dt
import numpy as np
import pandas as pd
import cv2
import sys
import queue
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from pathlib import Path
from datetime import datetime

# Add the parent folder.
sys.path.append('')

from convert_2Dto3D_tools.utils_marvin import setup_logger, load_json
from convert_2Dto3D_tools.pointcloud_utils import image2classes, save_df_pointcloud
from convert_2Dto3D_tools.convert_marvin_pointclouds import load_marvin_calibration
from convert_2Dto3D_tools.reprojection_opencv import main_reproject
from convert_2Dto3D_tools.filter_pcd_with_upsampling import OctreeFilter


class Repojection():
    def __init__(self, config_dict, logger):
        # Note config dict is already an multiprocess safe variable!
        self.config_dict = config_dict
        self.logger = logger
        # X,Y, Z location of the camera's relative to the origin of the pointcloud
        self.obj = load_marvin_calibration(Path(r"camera_params_marvin"))
        # self.add_extra_cams = self.config_dict.get("add_extra_cams", False)
        self.cam_numbers = self.config_dict.get("cam_numbers")
        self.cam_columns = []
        for x in self.cam_numbers:
            self.cam_columns.append("x"+str(x))
            self.cam_columns.append("y"+str(x))
        self.add_extra_cams=False
        if max(self.cam_numbers)>14:
            self.add_extra_cams=True
            self.obj.create_new_poses()
        self.cam_locations = self.obj.get_mmo_array_cams(self.cam_numbers)
        self.coordinates_csvfile = None

        # self.cam_locations = np.array([
        #     [-766, -88, 1213],
        #     [188, -862, 1198],
        #     [1215, -189, 1168],
        #     [893, 996, 1161],
        #     [-326, 1057, 1180],
        #     [-764, -96, 450],
        #     [177, -847, 443],
        #     [1180, -191, 414],
        #     [868, 973, 403],
        #     [-329, 1038, 432],
        #     [-790, -109, -281],
        #     [160, -868, -289],
        #     [1180, -201, -326],
        #     [861, 974, -334],
        #     [-353, 1037, -305]
        # ])

    def color_pc(self, data_folder):
        """ Takes a  *Coordinates-LIB.csv as input
            Based on the camera position, it computes the point to pixel translation
            Using the colors in the images and clases in the DL output the pointcloud is colored
        """

        # coordinates_csvfile = list(data_folder.glob("*pointcloud_Coordinates-Skin-LIB.csv"))[0]
        self.coordinates_csvfile = list(data_folder.glob("*.csv"))[0]

        # Load CSV and add header
        coordinates_dt = dt.fread(str(self.coordinates_csvfile))[:, :3] 
        coordinates_df = coordinates_dt.to_pandas()*1000 ## convert to mm
        coordinates_df.columns = ['x', 'y', 'z']
        coordinates_df[["blue", "green", "red"]]= [0, 0, 0]
        # if self.num_cams:
        cam_nums = [str(x) for x in self.cam_numbers]
        coordinates_df = main_reproject(self.obj, cam_nums, coordinates_df, unmirror=False)
        coordinates_df = coordinates_df[["x","y","z"]+self.cam_columns]
        # Create the input parameters for each camera
        camera_list = []
        # for i in range(0, self.num_cams):
        for i in self.cam_numbers:
            cam_i_df = coordinates_df.loc[:, ['x', 'y', 'z', f'x{i}', f'y{i}']]
            # cameraimages = data_folder.glob(f"*-cam_{i:02}.png")
            cameraimages = (data_folder.parent.parent / "images" / data_folder.name).glob(f"*-cam_{i:02}.png")
            cameraimages_inference = (data_folder.parent.parent / "inference" / data_folder.name).glob(f"*-cam_{i:02}.png")

            image_paths = {}
            # Get path to image from that camera
            # for imagename in cameraimages:
            #     imagename = str(imagename)
            #     if self.config_dict['DL_seg'] and "-preseg-" in imagename:
            #         image_paths['segmented'] = imagename
            #     elif self.config_dict["run_phd_bart"] and "-preseg-" in imagename:
            #         image_paths['segmented'] = Path(imagename).parent.parent / config_dict["phd_experiment_name"] / Path(imagename).name
            #         if not image_paths['segmented'].is_file():
            #             image_paths['segmented'] = str(Path(imagename).parent.parent / config_dict["phd_experiment_name"] / (Path(imagename).stem + "_coloured.png"))
            #         else:
            #             image_paths["segmented"] = str(image_paths["segmented"])
            #     else:
            #         image_paths['rgb'] = imagename
            # for the extra cams, cameraimages = [], therefore:
            if i>14:
                image_paths['rgb'] = None
                cameraimages_inference = (data_folder.parent.parent / "inference" / data_folder.name).glob(f"*-cam_{i:02}_coloured.png")
                image_paths["segmented"] = list(cameraimages_inference)[0]
            else:
                image_paths["rgb"] = list(cameraimages)[0]
                image_paths["segmented"] = list(cameraimages_inference)[0]
            camera_list.append((image_paths, cam_i_df, i))

        # all camera's are seperate, which means all 16 can be computed at the same time
        with ThreadPoolExecutor(4) as executor:
            result_dfs = list(executor.map(lambda p: self.process_one_camera(*p), camera_list))

        # Combine all pointclouds
        complete_pc_df = pd.concat(result_dfs)
        complete_pc_df[["x", "y", "z"]]=complete_pc_df[["x", "y", "z"]].astype(int)

        # The same point can be seen by multiple camera's, these duplicate labels have to be removed.
        # We chose for averaging, downside is the long computation time.
        # Of the tested pivot table, groupby: mode & value count this was the fastest solution
        # in case of ties the first class label is returned
        complete_pc_df = complete_pc_df.groupby(['x', 'y', 'z']).agg({'class': lambda s: pd.Series.mode(s).iloc[0],
                                                                      'red': 'mean',
                                                                      'green': 'mean',
                                                                      'blue': 'mean',
                                                                      'counts': 'mean'})
        complete_pc_df.reset_index(inplace=True)
        complete_pc_df = complete_pc_df.round({'red': 0, 'green': 0, 'blue': 0})
        complete_pc_df[["x", "y", "z"]] = complete_pc_df[["x", "y", "z"]]/1000

        return complete_pc_df[['x', 'y', 'z', 'blue', 'green', 'red', 'counts', 'class']]

    def process_one_camera(self, image_paths, cam_df, camera_id):
        """"Function is multiprocessed, watchout for conflicts with self. variables"""
        # Pixel to point mapping
        all_pixels_df = self.camera_mapping(cam_df, camera_id)
        camera_df = self.project_color_and_classes(image_paths, all_pixels_df, camera_id)
        return camera_df

    def camera_mapping(self, cam_df, camera_id):
        """uses uclidian distance to map the closest points to the pixel
        Input: path to the image, a dataframe for the camera with columns: x, y, z, z<camera>, y<camera> and the camera_id
        Output: An tuple with the camera ID and filtered dataframe with x, y, z coordinated and the matching pixels"""
        cam_df = cam_df.dropna()
        cam_df[f'x{camera_id}_s'] = (cam_df[f'x{camera_id}']/self.config_dict["camera_scale_factor"]).astype(int)
        cam_df[f'y{camera_id}_s'] = (cam_df[f'y{camera_id}']/self.config_dict["camera_scale_factor"]).astype(int)

        # Combine x and y column
        cam_df['xy_i'] = cam_df.loc[:, f'x{camera_id}_s'].astype(str) + " " + cam_df.loc[:, f'y{camera_id}_s'].astype(str)

        # Dont sort so the counts can be merged back easily
        value_counts = cam_df['xy_i'].value_counts(sort=False)
        cam_df["counts"] = cam_df["xy_i"].map(value_counts)

        # These pixels only translate to one point
        mapped_df = cam_df.loc[cam_df['counts'] == 1]

        # Pixels translate to multiple points
        to_map_df = cam_df.loc[cam_df['counts'] > 1].copy(deep=True)

        # Distanse translation camera 1: (x y z [mm])
        cam_loc = self.cam_locations[str(camera_id)]

        # Compute euclidian distance between points (x,y,z) and camera (x,y,z) location
        to_map_df[['x_dist', 'y_dist', 'z_dist']] = to_map_df[['x', 'y', 'z']].values - cam_loc
        to_map_df['distance'] = np.linalg.norm(to_map_df[['x_dist', 'y_dist', 'z_dist']].astype(float), axis=1)

        # index_list=to_map_df.loc[to_map_df.groupby('xy_1').distance.idxmin()] ## however not so fast
        # slightly faster according to
        # https://stackoverflow.com/questions/54470917/pandas-groupby-and-select-rows-with-the-minimum-value-in-a-specific-column
        min_value = to_map_df.groupby('xy_i')['distance'].min()
        to_map_df = to_map_df.merge(min_value, on='xy_i', suffixes=('', '_min'))
        to_map_df = to_map_df[to_map_df['distance'] == to_map_df['distance_min']]
        to_map_df.drop('distance_min', axis=1, inplace=True)

        all_pixels_df = pd.concat([mapped_df, to_map_df])
        return all_pixels_df

    def project_color_and_classes(self, image_paths, camera_df, camera_id):
        """Open the image and use the pixel to point conversion df to add colos the pointcloud"""
        
        segmented_img = cv2.imread(image_paths['segmented'])
        if np.all(segmented_img[:,:,0]==segmented_img[:,:,1]):
            class_binary_img = segmented_img[:,:, 0]
            # class_binary_img[class_binary_img==4] = 5
            class_binary_img[class_binary_img==255] = 0

        else:
            class_binary_img = image2classes(segmented_img, self.config_dict)

        # Assign classes
        camera_df['class'] = class_binary_img[camera_df[f'y{camera_id}'].values.astype(int), camera_df[f'x{camera_id}'].values.astype(int)]

        # Assign RGB colors
        if image_paths["rgb"] is not None:
            rgb_image = cv2.imread(image_paths['rgb'])
            camera_df['blue'] = rgb_image[camera_df[f'y{camera_id}'].values.astype(int), camera_df[f'x{camera_id}'].values.astype(int), 0]
            camera_df['green'] = rgb_image[camera_df[f'y{camera_id}'].values.astype(int), camera_df[f'x{camera_id}'].values.astype(int), 1]
            camera_df['red'] = rgb_image[camera_df[f'y{camera_id}'].values.astype(int), camera_df[f'x{camera_id}'].values.astype(int), 2]

        if self.config_dict["run_paper_settings"]:
            camera_df = camera_df[camera_df["class"] != self.config_dict['rgb_encoding']["Pot"]["class_id"]]
            return camera_df
        # Removing the Pot
        return camera_df

    def color_pcv2(self, xyz, img_list, img_seg_list):
        """ Takes a  *Coordinates-LIB.csv as input
            Based on the camera position, it computes the point to pixel translation
            Using the colors in the images and clases in the DL output the pointcloud is colored
        """

        # coordinates_csvfile = list(data_folder.glob("*pointcloud_Coordinates-Skin-LIB.csv"))[0]
        # self.coordinates_csvfile = list(data_folder.glob("*.csv"))[0]

        # Load CSV and add header
        # coordinates_dt = dt.fread(str(self.coordinates_csvfile))[:, :3] 
        # coordinates_df = coordinates_dt.to_pandas()*1000 ## convert to mm
        coordinates_df = pd.DataFrame(xyz*1000)
        coordinates_df.columns = ['x', 'y', 'z']
        coordinates_df[["blue", "green", "red"]]= [0, 0, 0]
        # if self.num_cams:
        cam_nums = [str(x) for x in self.cam_numbers]
        coordinates_df = main_reproject(self.obj, cam_nums, coordinates_df, unmirror=False)
        coordinates_df = coordinates_df[["x","y","z"]+self.cam_columns]
        # Create the input parameters for each camera
        camera_list = []

        img_list_dict = {}
        for x in img_list:
            key = str(int(x.stem.split("_")[-1]))
            img_list_dict[key]= x
        img_seg_dict = {}
        for x in img_seg_list:
            key = str(int(x.stem.split("_")[-1]))
            img_seg_dict[key]= x


        # for i in range(0, self.num_cams):
        for i in self.cam_numbers:
            cam_i_df = coordinates_df.loc[:, ['x', 'y', 'z', f'x{i}', f'y{i}']]
            # cameraimages = data_folder.glob(f"*-cam_{i:02}.png")
            # cameraimages = (data_folder.parent.parent / "images" / data_folder.name).glob(f"*-cam_{i:02}.png")
            # cameraimages_inference = (data_folder.parent.parent / "inference" / data_folder.name).glob(f"*-cam_{i:02}.png")

            image_paths = {}
            image_paths["rgb"] = img_list_dict[str(i)]
            image_paths["segmented"] = img_seg_dict[str(i)]

            # if i>14:
            #     image_paths['rgb'] = None
            #     cameraimages_inference = (data_folder.parent.parent / "inference" / data_folder.name).glob(f"*-cam_{i:02}_coloured.png")
            #     image_paths["segmented"] = list(cameraimages_inference)[0]
            # else:
            #     image_paths["rgb"] = list(cameraimages)[0]
            #     image_paths["segmented"] = list(cameraimages_inference)[0]
            camera_list.append((image_paths, cam_i_df, i))

        # all camera's are seperate, which means all 16 can be computed at the same time
        with ThreadPoolExecutor(4) as executor:
            result_dfs = list(executor.map(lambda p: self.process_one_camera(*p), camera_list))

        # Combine all pointclouds
        complete_pc_df = pd.concat(result_dfs)
        complete_pc_df[["x", "y", "z"]]=complete_pc_df[["x", "y", "z"]].astype(int)

        # The same point can be seen by multiple camera's, these duplicate labels have to be removed.
        # We chose for averaging, downside is the long computation time.
        # Of the tested pivot table, groupby: mode & value count this was the fastest solution
        # in case of ties the first class label is returned
        complete_pc_df = complete_pc_df.groupby(['x', 'y', 'z']).agg({'class': lambda s: pd.Series.mode(s).iloc[0],
                                                                      'red': 'mean',
                                                                      'green': 'mean',
                                                                      'blue': 'mean',
                                                                      'counts': 'mean'})
        complete_pc_df.reset_index(inplace=True)
        complete_pc_df = complete_pc_df.round({'red': 0, 'green': 0, 'blue': 0})
        complete_pc_df[["x", "y", "z"]] = complete_pc_df[["x", "y", "z"]]/1000

        return complete_pc_df[['x', 'y', 'z', 'blue', 'green', 'red', 'counts', 'class']]


class ReprojectionManager(Process):
    """"Connect and communicate with the Reprojection classes"""
    def __init__(self, config_dict, job_queue, shutdown_event, filter_queue):
        super(ReprojectionManager, self).__init__()
        self.config_dict = config_dict
        self.job_queue = job_queue
        self.filter_queue = filter_queue
        self.shutdown_event = shutdown_event

    def run(self):
        # Create model & logger in in the new process
        self.logger = setup_logger(name="Reprojection")
        self.reprojector = Repojection(self.config_dict, self.logger)
        self.logger.info("Reprojection started")
        while not self.shutdown_event.is_set():
            try:
                # Process single folder
                foldername = self.job_queue.get(block=True)
                self.logger.info(foldername)
                # Signal to stop worker
                if foldername is None:
                    self.filter_queue.put(foldername)
                    break

                # Only process new folders, unless rerunning
                reprojected_filename = Path.joinpath(foldername, 'reprojected_pointcloud.ply')
                if self.config_dict['force_reprocess'] or not Path.exists(reprojected_filename):
                    pcd = self.reprojector.color_pc(foldername)
                    save_df_pointcloud(reprojected_filename, pcd)

                self.cleanup_intermediates(foldername)
                # Put in next queue
                self.filter_queue.put(reprojected_filename)
            except queue.Empty:
                continue
            except Exception as exp:
                self.logger.error(exp, exc_info=True)

    def cleanup_intermediates(self, foldername):
        # Remove DLL output
        if self.config_dict['cleanup_DLL_output']:
            for imgfile in foldername.glob("*cam-pointcloud*.csv"):
                Path.unlink(imgfile)

        # Remove DL preseg images
        if self.config_dict['cleanup_DL_presegs']:
            for imgfile in foldername.glob("*preseg-cam*.png"):
                Path.unlink(imgfile)



def main_new_architecture(config_dict, camera_specs, xyz, img_list, img_seg_list, pc_file):
    logger = setup_logger(name="Reprojection")

    reprojector = Repojection(config_dict, logger)
    filter = OctreeFilter(config_dict, logger)

    reprojector.obj = camera_specs
    df = reprojector.color_pcv2(xyz, img_list, img_seg_list)
    # return df
    df = filter.filter_pcd(df, 
                      original_file=str(pc_file))
    return df


def main(config_dict, test_pots=["Harvest_02_PotNr_27"]):
    logger = setup_logger(name="Reprojection")

    # test_pots = ["Harvest_01_PotNr_179", "Harvest_01_PotNr_429", "Harvest_02_PotNr_27", "Harvest_02_PotNr_166", "Harvest_02_PotNr_240"]
    test_pots = ["Harvest_02_PotNr_27"]
    reprojector = Repojection(config_dict, logger)
    filter = OctreeFilter(config_dict, logger)

    for x in test_pots:
        folder = Path('example_data') / "annotations" / x
        a = datetime.now()
        pcd = reprojector.color_pc(folder)
        print(datetime.now()-a)

        # # Save pointcloud on disk
        save_folder = folder.parent / config_dict["phd_experiment_name"] / x
        # save_df_pointcloud(save_folder / ("reprojected_pointcloud" + config_dict["save_name_extension"]), pcd)
        
        # # optional do filtering directly as well
        filter.coordinates_csvfile = list(folder.glob("*.csv"))[0]
        df = filter.filter_pcd(save_folder / ("reprojected_pointcloud" + config_dict["save_name_extension"]), original_file=reprojector.coordinates_csvfile)
        print("Succesfuly saved 2Dto3D reprojection", save_folder)
        # save_df_pointcloud(save_folder / 'only_stemwork.ply', df)


if __name__ == '__main__':
    config_dir = Path("example_configs")
    config_dict = load_json(config_dir / "config_publication_experiment1.json")

    main(config_dict)

