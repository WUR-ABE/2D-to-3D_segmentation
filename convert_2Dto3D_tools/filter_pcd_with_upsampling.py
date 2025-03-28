""" Filters the leaves, stick, pot and other parts out of the pointcloud so only the stemwork remains as input for platscan3d"""
import numpy as np
import open3d as o3d
import sys
import queue
from pathlib import Path
from multiprocessing import Process
import pandas as pd
import datatable as dt

# Add the parent folder.
sys.path.append('')
from convert_2Dto3D_tools.utils_marvin import setup_logger, load_json
from convert_2Dto3D_tools.pointcloud_utils import save_df_pointcloud, load_df_pointcloud


class OctreeFilter():
    """ The pointcloud is converted to an Octree to enable quickly finding neighboring points.
    An octree consists of:
    - OctreeLeafNode: last node of an octree
        - OctreePointColorLeafNode: LeafNode with color information
        - OctreeInternalNode: Octree node with children
        - OctreeInternalPointNode: OctreeInternalNode with a list of point indices of its children
    for more information see: http://www.open3d.org/docs/latest/python_api/open3d.geometry.OctreeInternalNode.html
    """
    def __init__(self, config_dict, logger):
        self.config_dict = config_dict
        self.df = None
        # List of class id's to remove
        self.classes_to_remove = [config_dict['rgb_encoding']['Leaf']['class_id'],
                                  config_dict['rgb_encoding']['Pot']['class_id'],
                                  config_dict['rgb_encoding']['Pole']['class_id']]
        # config_dict['rgb_encoding']['background']['rgb_encoding']
        self.logger = logger

    # Colors RGB colors of classes > 1D array with classes!
    def filter_pcd(self, reprojected_file, original_file=None):
        """The goals is to remove the leaves, pot, stick:
        Endup with a pointcloud that only contains stemwork"""

        if isinstance(reprojected_file, str) or isinstance(reprojected_file, Path):
            self.df, pcd = load_df_pointcloud(reprojected_file, df_drop=[])
        else:
            self.df = reprojected_file
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.df[['x', 'y', 'z']].values)
            # if 'red' in self.df.columns and 'green' in self.df.columns and 'blue' in self.df.columns:
            #     pcd.colors = o3d.utility.Vector3dVector(self.df[['red', 'green', 'blue']].values / 255.0)
        self.df[["x", "y", "z"]] = self.df[["x", "y", "z"]]*1000
        self.df.reset_index(drop=True, inplace=True)
        self.df['new_class'] = self.df['class']
        # Create octree to improve segmentation accuracy using a majority vote algorithm
        octree = o3d.geometry.Octree(max_depth=7)
        octree.convert_from_point_cloud(pcd, size_expand=0.01)
        _ = octree.traverse(self.f_traverse)

        if self.config_dict['reconstructe_complete_pc']:
            self.reconstruct_complete_pc(reprojected_file, original_file)
        else:
            self.df_merged = self.df
            self.df_merged = self.df_merged.drop(columns=["counts", "class"])
            self.df_merged = self.df_merged.rename(columns={"new_class": "class"})
            self.df_merged["points_to_remove"] = 0

        self.df_merged[["x", "y", "z"]] = self.df_merged[["x", "y", "z"]]/1000

        # show_octree(octree)
        # save_df_pointcloud(str(reprojected_file.parent / 'reprojection_complete_octree_filtering.ply'), self.df_merged)
        if self.config_dict["run_paper_settings"]:
            save_df_pointcloud(str(reprojected_file.parent / (reprojected_file.parent.parent.name+'_reprojected_complete' + self.config_dict["save_name_extension"])),
           self.df_merged)

        # To extract the main stem and side stems we have to remove several items
        # First we remove the stick by fitting a cylinder
        if self.config_dict['subfilter_stick']:
            self.remove_stick()
        # Also remove pot and leaves
        # self.df_merged = self.df_merged[~self.df_merged['class'].isin(self.classes_to_remove)]
        # self.df_merged = self.df_merged[~self.df_merged['points_to_remove'].isin([1])]

        # self.df_merged.reset_index(drop=True, inplace=True)

        # If std< additional statistical filtering
        if self.config_dict["statistics_filtering"]:
            pcd_input = o3d.geometry.PointCloud()
            pcd_input.points = o3d.utility.Vector3dVector(self.df_merged[['x', 'y', 'z']].values)
            cl, ind = pcd_input.remove_statistical_outlier(nb_neighbors=20, std_ratio=.3)
            self.df_merged = self.df_merged.iloc[ind]

    
        return self.df_merged

    def f_traverse(self, node, node_info):
        """Custom open3d's traverse callback
        Used to create a list of nodeindexes to majority vote edge points to classes.
        """
        if isinstance(node, o3d.geometry.OctreeInternalNode):
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                pass
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):

                # Get uniqe values with the counts.
                classes, counts = np.unique(self.df.iloc[node.indices, self.df.columns.get_loc('class')], return_counts=True)
                # Majority vote assign class to node indices
                if np.all(counts == counts[0]):
                    selected_class = classes.min()
                    self.df.iloc[node.indices, self.df.columns.get_loc('new_class')] = selected_class
                else:
                    selected_class = classes[np.argmax(counts)]
                    self.df.iloc[node.indices, self.df.columns.get_loc('new_class')] = selected_class
        else:
            raise NotImplementedError('Node type not recognized!')

    def f_traverse_complete(self, node, node_info):
        """Custom open3d's traverse callback
        Used to create a list of nodeindexes to majority vote edge points to classes.
        """
        if isinstance(node, o3d.geometry.OctreeInternalNode):
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                pass
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            # last_node = node
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
                # Bart TODO: I believe you can always use fix_missing_pionts, when counts==0 it is skipped anyway
                # Get unique values with the counts.
                # BART TODO pottential speed up if no nan are detected
                classes, counts = np.unique(self.df_merged.iloc[node.indices, self.df_merged.columns.get_loc('new_class')],
                                            return_counts=True)
                counts = counts[~np.isnan(classes)]
                classes = classes[~np.isnan(classes)]

                if len(counts) == 0:
                    pass
                # Majority vote assign class to node indices
                elif np.all(counts == counts[0]):
                    selected_class = classes.min()
                    self.assign_classes(node, selected_class)
                else:
                    selected_class = classes[np.argmax(counts)]
                    self.assign_classes(node, selected_class)

        else:
            raise NotImplementedError('Node type not recognized!')

    def reconstruct_complete_pc(self, reprojected_file, original_file=None):
        """Because not all points are classifed through reprojection.
        We upsample the assigned classes (from the reprojection self.df) to the compleete
         point cloud (self.df_complete) using octree-based traversal.
        This octree traversal is done twice, first with a small tree and later with a larger tree size.

        Args:
        - self: Instance of the class containing the method.
        - reprojected_file: Path to the reprojected CSV file containing all coordinates.
        Returns:
        - None: Modifies the 'df_merged' DataFrame in the class instance.

        Note:
        - Performs octree traversal twice, first for all points and then for unmatched points,
          removing unmatched points afterwards.
        - Result of algorithm stored in 'class' after processing.
        """

        # Load complete point cloud
        if original_file is None:
            try:
                self.coordinates_csvfile =  list(reprojected_file.parent.rglob("*pointcloud_Coordinates-Skin-LIB.csv"))[0]
            except IndexError:
                self.coordinates_csvfile = list(reprojected_file.parent.parent.rglob("*pointcloud_Coordinates-Skin-LIB.csv"))[0]
        else: 
            self.coordinates_csvfile = original_file

        self.df_complete = dt.fread(str(self.coordinates_csvfile))
        self.df_complete = self.df_complete[:, :3].to_pandas()*1000 ## convert to mm

        self.df_complete.columns = ['x', 'y', 'z']
        
        self.df_merged = pd.merge(self.df_complete, self.df[['x', 'y', 'z', 'blue', 'green', 'red', "new_class"]],
                                  on=['x', 'y', 'z'], how="left")
        # There are two ways to run the algorithm, or assign all values already, this could speed up the system!
        self.df_merged["new_class_complete"] = np.nan
        # self.df_merged["new_class_complete"]=self.df_merged["new_class"].copy(deep=True)

        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(self.df_complete[['x', 'y', 'z']].values)
        # First one run with similar octree setting
        octree_complete = o3d.geometry.Octree(max_depth=7)
        octree_complete.convert_from_point_cloud(pcd_input, size_expand=0.01)
        _ = octree_complete.traverse(self.f_traverse_complete)
        self.logger.info("unmatched points remaining %d" % sum(np.isnan(self.df_merged.new_class_complete)))

        # Another run for only the unmatched points,(these are unmatched because in the leaf of the octree no clases were found)
        # of course this could be repated for max_depth 5,4,3 etc as well.
        octree_complete = o3d.geometry.Octree(max_depth=6)
        octree_complete.convert_from_point_cloud(pcd_input, size_expand=0.01)
        _ = octree_complete.traverse(self.f_traverse_complete)
        self.logger.info("unmatched points remaining, will be removed%d" % sum(np.isnan(self.df_merged.new_class_complete)))

        # Remove thos points
        self.df_merged = self.df_merged[self.df_merged.new_class_complete != np.nan]
        self.df_merged = self.df_merged[~np.isnan(self.df_merged.new_class_complete)]

        self.df_merged.drop(["new_class"], axis=1, inplace=True)
        # self.df_merged = self.df_merged[~self.df_merged['new_class'].isin([0])]
        # Remove background (will only occur for bolai data)
        self.df_merged["points_to_remove"] = 0
        self.df_merged.reset_index(drop=True, inplace=True)
        self.df_merged = self.df_merged.rename(columns={"new_class_complete": "class"})

    def remove_stick(self):
        """"Remove stick by filtering a line through the pole points.
        Uses the 'fit_line' method to fit a line through points classified as poles and determines distances to identify sticks.
        Assigns a value of 1 to the 'points_to_remove' column for points identified as sticks.

        Returns:
        - None: Modifies the 'points_to_remove' column in the DataFrame of the class instance.
        """
        all_points = self.df_merged[['x', 'y', 'z']]._values
        stick_indices = self.df_merged.index[self.df_merged['class'] ==
                                             self.config_dict['rgb_encoding']['Pole']['class_id']].tolist()
        pcd_fit_cylinder = all_points[stick_indices]

        # Fit line through points classified as pole
        status, vector_up_down, xyz_start, xyz_end, _ = self.fit_line(pcd_fit_cylinder)
        # Determine distance orthogonal to vector and points. If distance is within threshold assign as stick
        # TODO maybe fit line twice?
        if status == 'ok':
            all_distance = np.linalg.norm(np.abs(np.cross(vector_up_down,
                                                          np.subtract(xyz_start, all_points)))/np.linalg.norm(vector_up_down),
                                          axis=1)
            # To remove stick
            self.df_merged.iloc[all_distance < 6, self.df_merged.columns.get_loc('points_to_remove')] = 1

    def fit_line(self, xyz):
        """Function that receives an Nx3 np array and fits a line using the best vector
        input
        -----
        xyz: numpy (float) Nx3 array with N= xyz coordinates
        -------
        returns
        vector_up_down: a vector that that desribes the line
        xyz_start:      start punt of vector
        xyz_end:        end punt of vector
        xyz_mean:       mean xyz point of xyz
        """
        xyz_mean = xyz.mean(axis=0)
        A = xyz - xyz_mean

        # Use Singular Value Decomposition
        # to reduce summarize data into three vectors (vh) select first most important vector
        # [U,S,vh] = np.linalg.svd(A.T,full_matrices=False)
        _, _, vh = np.linalg.svd(A, False)
        try:
            d = vh[0]
            # To make sure start is highest point and end is lowest point
            if d[2] > 0:
                d = d * -1
        except IndexError:
            return 'noise', [], [], [], []

        tr = np.dot(d, A.T)
        tr1 = tr.min()
        tr2 = tr.max()
        xyz_start = xyz_mean + np.dot(tr1, d)
        xyz_end = xyz_mean + np.dot(tr2, d)
        vector_up_down = xyz_end - xyz_start

        return 'ok', vector_up_down, xyz_start, xyz_end, xyz_mean

    def assign_classes(self, node, selected_class):
        """This method assigns the given selected class label to non-matched points
        in the DataFrame associated with the OctreeNode.
        input:
        -----
        - node (OctreeNode): The OctreeNode representing a node in the octree.
        - selected_class (int): The class label to be assigned.
        returns:
        -------
        Nothing it only updates self.df_merged
        """
        # Only assign selected class to non matched points,
        # By making this subselection is poossible to re-use f_traverse_complete with different octree sizes
        selected_row = self.df_merged.iloc[node.indices, self.df_merged.columns.get_loc('new_class_complete')]
        # value_condition = selected_row==np.nan
        value_condition = np.isnan(selected_row)
        subselected_rows = selected_row[value_condition]
        # print(self.df_merged.loc[selected_row.index])
        if len(subselected_rows) > 0:
            self.df_merged.loc[subselected_rows.index, "new_class_complete"] = selected_class
        # For colours we do the same trick
        selected_row = self.df_merged.iloc[node.indices][["blue", "green", "red"]]
        value_condition = np.isnan(selected_row["blue"])
        subselected_rows = selected_row[value_condition]
        if len(subselected_rows) > 0:
            self.df_merged.loc[subselected_rows.index, ["blue", "green", "red"]] = selected_row.mean().astype(int).values


class FilterManager(Process):
    """"Connect and communicate with the Reprojection classes"""
    def __init__(self, config_dict, job_queue, shutdown_event, plantscan_queue):
        super(FilterManager, self).__init__()
        self.config_dict = config_dict
        self.job_queue = job_queue
        self.plantscan_queue = plantscan_queue
        self.shutdown_event = shutdown_event

    def run(self):
        # Create model & logger in in the new process
        self.logger = setup_logger(name="Filtering")
        self.filter = OctreeFilter(self.config_dict)
        self.logger.info("Filter process started")
        while not self.shutdown_event.is_set():
            try:
                # Process single file
                filename = self.job_queue.get(block=True)
                self.logger.info(filename)
                # Signal to stop worker
                if filename is None:
                    self.plantscan_queue.put(filename)
                    break

                # Only process new folders, unless rerunning
                filtered_filename = Path.joinpath(filename.parent, 'only_stemwork.ply')
                if self.config_dict['force_reprocess'] or not Path.exists(filtered_filename):
                    df = self.filter.filter_pcd(filename)
                    save_df_pointcloud(filtered_filename, df)

                # Put in next queue
                self.plantscan_queue.put(filename)
            except queue.Empty:
                continue
            except Exception as exp:
                self.logger.error(exp, exc_info=True)


if __name__ == '__main__':
    cfd = Path(__file__).parent.parent.resolve()
    config_dict = load_json(Path('config.json'))
    logger = setup_logger(name="filer_pcd_with_upsampling")

    folder = Path('/home/agro/w-drive-vision/GARdata/experiments/marvin_pointcloud/data_bolai/Raw_Images')

    config_dict = load_json(Path('config_bolai_data.json'))

    test_pots = ["Harvest_01_PotNr_179", "Harvest_01_PotNr_429", "Harvest_02_PotNr_27", "Harvest_02_PotNr_166", "Harvest_02_PotNr_240"]
    test_pots = ["Harvest_02_PotNr_27_backup"]
    filter = OctreeFilter(config_dict, logger)

    for x in test_pots:
        # folder = Path('/local/marvin_testdata/data_bolai/Raw_Images') / x / 'Data'
        input_folder = folder / x / "Data"
        df = filter.filter_pcd(input_folder / "reprojected_pointcloud.ply")
        save_df_pointcloud(input_folder / 'only_stemwork.ply', df)
