# import math
# import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import build_dataloader
from losses import CombinedLoss
from models import build_network
# DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
# from models.LCCNet import LCCNet
from quaternion_distances import quaternion_distance
from utils.utils import cfg_from_yaml_file

from workspace.targetless_calib.datasets.kitti_odom import \
    DatasetLidarCameraKittiOdometry
# from tensorboardX import SummaryWriter
from workspace.targetless_calib.utils.utils import \
    rotate_back  # merge_inputs,; overlay_imgs,; quat2mat,; rotate_forward,; tvector2mat,

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False


def main():
    cfg = cfg_from_yaml_file()
    # params
    epochs = cfg.OPTIMIZATION.NUM_EPOCHS
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    output_dir = Path(cfg.DATA_CONFIG.OUTPUT_DIR).resolve()
    ckpt_dir = output_dir / "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_set, train_loader, train_sampler = build_dataloader(
        cfg.DATA_CONFIG, batch_size=batch_size, training=True
    )

    model = build_network(cfg.MODEL, dataset=train_set)
    # feat = 1
    # md = 4
    # model = LCCNet(
    #     input_size,
    #     use_feat_from=feat,
    #     md=md,
    #     use_reflectance=config["use_reflectance"],
    #     dropout=config["dropout"],
    #     Action_Func="leakyrelu",
    #     attention=False,
    #     res_num=18,
    # )
    loss_fn = CombinedLoss(
        cfg.RESCALE_TRANSLATION,
        cfg.RESCALE_ROTATION,
        cfg.WEIGHT_POINTCLOUD,
    )
    model = model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    # Probably this scheduler is not used
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 50, 70], gamma=0.5
    )
    # rescale_transl
    # config["rescale_transl"],
    # config["rescale_rot"],
    # config["weight_point_cloud"],

    # DatasetLidarCameraKittiOdometry
    # img_shape = (384, 1280)
    # input_size = (256, 512)
    # config["checkpoints"] = os.path.join(config["checkpoints"], _conf"])

    # dataset_train = dataset_class(
    #     config["data_folder"],
    #     max_r=config["max_r"],
    #     max_t=config["max_t"],
    #     split="train",
    #     use_reflectance=config["use_reflectance"],
    #     val_sequence=config["val_sequence"],
    # )
    # dataset_val = dataset_class(
    #     config["data_folder"],
    #     max_r=config["max_r"],
    #     max_t=config["max_t"],
    #     split="val",
    #     use_reflectance=config["use_reflectance"],
    #     val_sequence=config["val_sequence"],
    # )


    # loss function choice
    # if config["loss"] == "simple":
    #     loss_fn = ProposedLoss(config["rescale_transl"], config["rescale_rot"])
    # elif config["loss"] == "geometric":
    #     loss_fn = GeometricLoss()
    #     loss_fn = loss_fn.cuda()
    # elif config["loss"] == "points_distance":
    #     loss_fn = DistancePoints3D()
    # elif config["loss"] == "L1":
    #     loss_fn = L1Loss(config["rescale_transl"], config["rescale_rot"])
    # elif config["loss"] == "combined":

    # else:
    #     raise ValueError("Unknown Loss Function")

    # runs = datetime.now().strftime('%b%d_%H-%M-%S') + "/"
    # train_writer = SummaryWriter('./logs/' + runs)
    # ex.info["tensorflow"] = {}
    # ex.info["tensorflow"]["logdirs"] = ['./logs/' + runs]

    # network choice and settings
    # if config["network"].startswith("Res"):
    #     feat = 1
    #     md = 4
    #     split = config["network"].split("_")
    #     for item in split[1:]:
    #         if item.startswith("f"):
    #             feat = int(item[-1])
    #         elif item.startswith("md"):
    #             md = int(item[2:])
    #     assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
    #     assert 0 < md, "md must be positive"
    # Res_f1

    # else:
    #     raise TypeError("Network unknown")
    if config["weights"] is not None:
        print(f"Loading weights from {config['weights']}")
        checkpoint = torch.load(config["weights"], map_location="cpu")
        saved_state_dict = checkpoint["state_dict"]
        model.load_state_dict(saved_state_dict)

        # original saved file with DataParallel
        # state_dict = torch.load(model_path)
        # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)

    # model = model.to(device)
    model = nn.DataParallel(model)
    model = model.cuda()

    # print(
    #     "Number of model parameters: {}".format(
    #         sum([p.data.nelement() for p in model.parameters()])
    #     )
    # )

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # if _config["loss"] == "geometric":
    #     parameters += list(loss_fn.parameters())
    # if config["optimizer"] == "adam":
    optimizer = optim.Adam(
        parameters, lr=config["BASE_LEARNING_RATE"], weight_decay=5e-6
    )
    # Probably this scheduler is not used
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 50, 70], gamma=0.5
    )
    # else:
    #     optimizer = optim.SGD(
    #         parameters,
    #         lr=_config["BASE_LEARNING_RATE"],
    #         momentum=0.9,
    #         weight_decay=5e-6,
    #         nesterov=True,
    #     )

    starting_epoch = config["starting_epoch"]
    # if config["weights"] is not None and config["resume"]:
    #     checkpoint = torch.load(config["weights"], map_location="cpu")
    #     opt_state_dict = checkpoint["optimizer"]
    #     optimizer.load_state_dict(opt_state_dict)
    #     if starting_epoch != 0:
    #         starting_epoch = checkpoint["epoch"]

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.0
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    for epoch in range(starting_epoch, config["epochs"] + 1):
        # EPOCH = epoch
        print("This is %d-th epoch" % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.0
        # if _config["optimizer"] != "adam":
        #     _run.log_scalar(
        #         "LR",
        #         _config["BASE_LEARNING_RATE"] * math.exp((1 - epoch) * 4e-2),
        #         epoch,
        #     )
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = _config["BASE_LEARNING_RATE"] * math.exp(
        #             (1 - epoch) * 4e-2
        #         )
        # else:
        #     # scheduler.step(epoch%100)
        #     _run.log_scalar("LR", scheduler.get_lr()[0])

        ## Training ##
        time_for_50ep = time.time()
        for batch_idx, sample in enumerate(TrainImgLoader):
            # print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample["tr_error"] = sample["tr_error"].cuda()
            sample["rot_error"] = sample["rot_error"].cuda()

            start_preprocess = time.time()
            for idx in range(len(sample["rgb"])):
                # ProjectPointCloud in RT-pose
                real_shape = [
                    sample["rgb"][idx].shape[1],
                    sample["rgb"][idx].shape[2],
                    sample["rgb"][idx].shape[0],
                ]

                sample["point_cloud"][idx] = sample["point_cloud"][
                    idx
                ].cuda()  # pointcloud pose in camera plane
                pc_lidar = sample["point_cloud"][idx].clone()

                if config["max_depth"] < 80.0:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < config["max_depth"]].clone()

                depth_gt, uv = lidar_project_depth(
                    pc_lidar, sample["calib"][idx], real_shape
                )  # image_shape
                depth_gt /= config["max_depth"]

                R = mathutils.Quaternion(sample["rot_error"][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample["tr_error"][idx])
                RT = T * R

                pc_rotated = rotate_back(
                    sample["point_cloud"][idx], RT
                )  # Pc` = RT * Pc

                if config["max_depth"] < 80.0:
                    pc_rotated = pc_rotated[
                        :, pc_rotated[0, :] < config["max_depth"]
                    ].clone()

                depth_img, uv = lidar_project_depth(
                    pc_rotated, sample["calib"][idx], real_shape
                )  # image_shape
                depth_img /= config["max_depth"]

                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                rgb = sample["rgb"][idx].cuda()
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = img_shape[0] - rgb.shape[1]  # // 2
                shape_pad[1] = img_shape[1] - rgb.shape[2]  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)

                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
            end_preprocess = time.time()
            loss, R_predicted, T_predicted = train(
                model,
                optimizer,
                rgb_input,
                lidar_input,
                sample["tr_error"],
                sample["rot_error"],
                loss_fn,
                sample["point_cloud"],
                config["loss"],
            )

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            # if batch_idx % config["log_frequency"] == 0:
            #     show_idx = 0
            #     # output image: The overlay image of the input rgb image
            #     # and the projected lidar pointcloud depth image
            #     rotated_point_cloud = pc_rotated_input[show_idx]
            #     R_predicted = quat2mat(R_predicted[show_idx])
            #     T_predicted = tvector2mat(T_predicted[show_idx])
            #     RT_predicted = torch.mm(T_predicted, R_predicted)
            #     rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

            #     depth_pred, uv = lidar_project_depth(
            #         rotated_point_cloud,
            #         sample["calib"][show_idx],
            #         real_shape_input[show_idx],
            #     )  # or image_shape
            #     depth_pred /= config["max_depth"]
            #     depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

            #     pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
            #     input_show = overlay_imgs(
            #         rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0)
            #     )
            #     gt_show = overlay_imgs(
            #         rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0)
            #     )

            #     pred_show = torch.from_numpy(pred_show)
            #     pred_show = pred_show.permute(2, 0, 1)
            #     input_show = torch.from_numpy(input_show)
            #     input_show = input_show.permute(2, 0, 1)
            #     gt_show = torch.from_numpy(gt_show)
            #     gt_show = gt_show.permute(2, 0, 1)

            local_loss += loss["total_loss"].item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print(
                    f"Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, "
                    f"time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, "
                    # f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
                    f"time for 50 iter: {time.time()-time_for_50ep:.4f}"
                )
                time_for_50ep = time.time()
                # _run.log_scalar("Loss", local_loss / 50, train_iter)
                local_loss = 0.0
            total_train_loss += loss["total_loss"].item() * len(sample["rgb"])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print("------------------------------------")
        print(
            "epoch %d total training loss = %.3f"
            % (epoch, total_train_loss / len(dataset_train))
        )
        print("Total epoch time = %.2f" % (time.time() - epoch_start_time))
        print("------------------------------------")

       