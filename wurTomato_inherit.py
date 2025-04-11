from pathlib import Path
# import os
# import argparse
import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt

# from torch.utils.data import Dataset
# from tqdm import tqdm
# import requests
# from zipfile import ZipFile
# from pathlib import Path
import pandas as pd
# import natsort
import argparse

from convert_2Dto3D_tools.utils_marvin import load_json
from convert_2Dto3D_tools.pointcloud_utils import load_df_pointcloud
import convert_2Dto3D_tools.reprojection_paper as reprojection
from convert_2Dto3D_tools.reprojection_paper import main_new_architecture

from TomatoWUR import wurTomato
from TomatoWUR.scripts import visualize_examples as ve
import subprocess


class wurTomato_2dto3d(wurTomato.WurTomatoData):

    config_dict = load_json(Path("example_configs") / "config_publication_experiment1.json")
    # config_dict = load_json(Path("example_configs") / "config_publication_experiment4.json")
    

    def convert2Dto3D(self, index=2):

        img_list, img_seg_list = self.get_2d_images(index)
        self.load_camera_specs()
        # self.cfg["camera_scale_factor"] = 1
        # self.cfg["cam_numbers"] = [0,1,2,3,4]

        xyz = self.load_xyz_array(index)#*[-1, 1,1] - [400, 0,0 ]
        new_df = main_new_architecture(config_dict=self.cfg, 
                              camera_specs=self.camera_specs,
                              xyz=xyz,
                              img_list=img_list,
                              img_seg_list=img_seg_list,
                              pc_file=self.dataset[index]["file_name"],
                              )
        new_df["class"] = new_df["class"].astype(int)
        colors_classes = wurTomato.rgb_array[new_df["class"].values]
        ve.vis(pc = new_df[["x", "y", "z"]], colors=colors_classes)

    def visualise_output_2Dto3D(self, txt_name="Harvest_02_PotNr_27"):
        filename = Path("example_data") / "output" / txt_name / "output_reprojected_complete.ply"

        df, _ = load_df_pointcloud(filename, return_pointcloud=False, df_drop=[ "red", "green", "blue"])
        df.loc[df["class"]==5, "class"] = 4
        colors_classes = wurTomato.rgb_array[df["class"].astype(int).values]
        ve.vis(pc = df[["x", "y", "z"]], colors=colors_classes)


    def visualise_output_3Dalgorithm(self, txt_name="Harvest_02_PotNr_27"):
        """Visualisation of Point Transformer algorithm."""
        file_name = Path("example_data") / "annotations" / txt_name / (txt_name + ".csv")

        df = pd.read_csv(file_name, low_memory=False)
        pred_name = Path("example_data") / "output_ptv3" / "result" / (txt_name + "_pred.npy")
        labels = np.load(str(pred_name)) + 1 # + 1 because pointcept library starts _at 0
        colors_classes = wurTomato.rgb_array[labels]
        ve.vis(pc = df[["x", "y", "z"]], colors=colors_classes)

    def train(self):
        print("please run ./train_tomatoWUR.sh")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process 2D to 3D conversion and visualization.")
    parser.add_argument("--convert", action="store_true", help="Run the 2D to 3D conversion.")
    parser.add_argument("--visualise_2dto3d", action="store_true", help="Visualize the output of 2D to 3D conversion.")
    parser.add_argument("--visualise_ptv3", action="store_true", help="Visualize the output of the 3D algorithm.")
    parser.add_argument("--txt_name", type=str, default="Harvest_02_PotNr_27", help="Name of the text file for visualization.")
    parser.add_argument("--train",action="store_true", help="Training the ptv3")


    args = parser.parse_args()

    obj = wurTomato_2dto3d()
    # obj.convert2Dto3D()


    if args.convert:
        obj.convert2Dto3D()
    if args.visualise_2dto3d:
        obj.visualise_output_2Dto3D(txt_name=args.txt_name)
    if args.visualise_ptv3:
        obj.visualise_output_3Dalgorithm(txt_name=args.txt_name)