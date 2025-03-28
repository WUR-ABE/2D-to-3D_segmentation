

# About 
Official implementation of the paper:

**3D plant segmentation: comparing a 2D-to-3D segmentation method with state-of-the-art 3D segmentation algorithms**



# 3D tomato dataset


This repo contains two items. 
1. A submodule related to the data of this paper. The class that can be used to visualize the TomatoWUR. The data will be automatically downloaded by running python wurTomato_inherit.py
2. An example how to apply the 2D-to-3D reprojection method assuming that you already have segmented the images using Mask2Former for example.
3. An example to use the dataset to train a 3D semantic segmentation algorithm using the pointcept git.  

<center>
    <p align="center">
        <img src="Resources/3D_tomato_plant.png" height="300" />
        <img src="Resources/3D_tomato_plant_semantic.png" height="300" />
    </p>
</center>

## Installation



Git clone our repo including submodules:
```
git clone https://github.com/WUR-ABE/2D-to-3D_segmentation
cd 2D-to-3D_segmentation
git submodule update --init --recursive
```

** Data visualisation
conda create --name 2dto3d python==

**Training
For training we recommend the docker. Note that the visualisation does not work for devcontainer.
Make sure your docker environment containts the nvidia docker to get acces to your gpu. It can take a up to 2 hours to install (because of flash attention installation is time intensive)

```
docker compose build
```

## Download and view dataset
To download the dataset, run python wurTomato_inherit.py

If everything is correct a folder in the TomatoWUR git will be created.
If not, then download the dataset by hand using following [link](https://data.4tu.nl/ndownloader/items/e2c59841-4653-45de-a75e-4994b2766a2f/versions/1). Create a folder named TomatoWUR/data/ and unzip results overhere.


## Run 2D-to-3D reprojection
Following line will run the 2D to 3D reprojection method:
```
python wurTomato.py --convert
```

## Run 2Dto3D and point cept visualisation
Following line will run the 2D to 3D reprojection method:
```
python wurTomato.py --visualise_2dto3d
python wurTomato.py --visualise_ptv3
```


## Voxel-carving / shape-from-silhouette
In the paper the 3D point clouds are made using the MaxiMarvin setup in NPEC (Wageningen University and Research).
The code for the MaxiMarvin is not available. However, to test the proof of concept please have a look at the TomatoWUR\Wurtomato.py -> voxel_carving function



## Training a PointCept:
Training a semantic segmenation algorithm is done using the json in the dataset folder. See example below. (default training without pre-trained weights).

```
train_tomatoWUR.sh
```

## Inference Pointcept
This will run the algorithm with pt3 from paper. Saves the prediction in a npy file in the save_path + result folder. Weights are available on request.

```
python Pointcept/tools/test.py --config-file example_configs/semseg-pt-v3m1-0-base.py --num-gpus 1 --options weight=example_configs/20240516_2022_ptv3_pretrained_default_lr_model_best.pth save_path=example_data/output_ptv3/

```


## Acknowledgement
This github would not be possible without open acces of several important libraries. Many credits to those librabies.

- Pointcept:              https://github.com/Pointcept/Pointcept
- Swin3D:                 https://github.com/microsoft/Swin3D
- TomatoWUR dataset:      https://github.com/orgs/WUR-ABE/repositories/tomatowur

For questions related to paper or code please send an email to bart.vanmarrewijk@wur.nl or open a git issue
