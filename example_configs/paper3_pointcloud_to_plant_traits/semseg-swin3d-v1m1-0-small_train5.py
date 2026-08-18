_base_ = ["../../Pointcept/configs/_base_/default_runtime.py"]
# import os
# import json
# misc custom setting
batch_size = 4  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True

# # dataset settings
# dataset_type = "MarvinDataset"
# data_root = os.environ.get("dataset_folder",None) + "/datasets/" + os.environ.get("dataset_name",None) #+ "/images/"
# train_name = data_root + os.environ.get("ann_version_train", "/train.json") #+ "/train.json"
# val_name = data_root + os.environ.get("ann_version", None) + "/val.json"
# test_name = data_root + os.environ.get("ann_version", None) + "/test.json"
# # temp = "/home/agro/w-drive-vision/GARdata/datasets/marvin_pointcloud/anns/2_20240324_correct/test.json"

# print("data_root=",data_root)

# f = open(str(os.environ.get("dataset_folder",None) + "/datasets/" + os.environ.get("dataset_name",None) + "/metadata.json"), "r")
# data = json.load(f)
# classes = tuple(data["classes"])
# del os
# del f
# del json
dataset_type = "TomatoWURCSV"
data_root = "TomatoWUR/data/TomatoWUR/ann_versions/0-paper-2Dto3D/json/"
data_root= "/media/agro/PhDBart1/2D-to-3D_segmentation/TomatoWUR/data/TomatoWUR/ann_versions/0-paper-2Dto3D/json/"

k_fold = 5
train_name = data_root + f"train_{k_fold}.json"
val_name = data_root + f"test_{k_fold}.json"
test_name = data_root + f"test_{k_fold}.json"
classes = ["leaves", "main_stem", "pole", "side_stem"]

grid_size = 0.002

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="Swin3D-v1m1",
        in_channels=9,
        num_classes=len(classes),
        base_grid_size=grid_size,
        depths=[2, 4, 9, 4, 4],
        channels=[48, 96, 192, 384, 384],
        num_heads=[6, 6, 12, 24, 24],
        window_sizes=[5, 7, 7, 7, 7],
        quant_size=4,
        drop_path_rate=0.3,
        up_k=3,
        num_layers=5,
        stem_transformer=True,
        down_stride=3,
        upsample="linear_attn",
        knn_down=True,
        cRSE="XYZ_RGB_NORM",
        fp16_mode=1,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=254)],
)

def_lr = 0.006

# scheduler settings
epoch = 600
optimizer = dict(type="AdamW", lr=def_lr, weight_decay=0.05)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[def_lr, def_lr/10],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=def_lr/10)]

data = dict(
    num_classes=len(classes),
    ignore_index=254,
    names=classes,
    train=dict(
        # type=dataset_type,
        # split="train",
        # data_root=data_root,
        type=dataset_type,
        lr_file = train_name,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                # grid_size=0.02,
                grid_size=grid_size, ## added bart was 0.02
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_displacement=True,
            ),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal", "displacement"),
                coord_feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        # split="val",
        # data_root=data_root,
        lr_file = val_name,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                # grid_size=0.02,
                grid_size=grid_size, ## added bart was 0.02
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_displacement=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal", "displacement"),
                coord_feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        # split="val",
        # data_root=data_root,
        lr_file = test_name,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                # grid_size=0.02,
                grid_size=grid_size, ## added bart was 0.02
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_displacement=True,
                keys=("coord", "color", "normal"),
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal", "displacement"),
                    coord_feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[0],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.95, 0.95]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.95, 0.95]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.95, 0.95]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.95, 0.95]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[0],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.05, 1.05]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.05, 1.05]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.05, 1.05]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.05, 1.05]),
                # ],
                # [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
