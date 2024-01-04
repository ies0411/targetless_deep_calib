# import math
# import os
# import random
# import time
import argparse
from pathlib import Path

# import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from datasets import build_dataloader
from models import build_network
from opt.loss import CombinedLoss

# DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
# from models.LCCNet import LCCNet
# from quaternion_distances import quaternion_distance
from utils.train_utils import train_model
from utils.utils import cfg_from_yaml_file

# from tensorboardX import SummaryWriter
# from workspace.targetless_calib.utils.utils import (
#     rotate_back,
# )  # merge_inputs,; overlay_imgs,; quat2mat,; rotate_forward,; tvector2mat,

# from workspace.targetless_calib.datasets.kitti_odom import (
#     DatasetLidarCameraKittiOdometry,
# )


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )
    args = parser.parse_args()
    cfg = cfg_from_yaml_file(args.cfg_file)

    return args, cfg


# TODO : rotation library check
def main():
    args, cfg = parse_config()
    print("===load config===")
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
    print("===build model===")
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
    # if config["weights"] is not None:
    #     print(f"Loading weights from {config['weights']}")
    #     checkpoint = torch.load(config["weights"], map_location="cpu")
    #     saved_state_dict = checkpoint["state_dict"]
    #     model.load_state_dict(saved_state_dict)

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

    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # optimizer = optim.Adam(
    #     parameters, lr=cfg.BASE_LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    # )
    # # Probably this scheduler is not used
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[20, 50, 70], gamma=0.5
    # )
    start_epoch = it = 0
    # TODO : load checkpoints
    train_model(
        model,
        optimizer,
        train_loader,
        lr_scheduler=scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=epochs,
        start_iter=it,
        # rank=cfg.LOCAL_RANK,
        # tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        # train_sampler=train_sampler,
        # lr_warmup_scheduler=lr_warmup_scheduler,
        # ckpt_save_interval=args.ckpt_save_interval,
        # max_ckpt_save_num=args.max_ckpt_save_num,
        # merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        # logger=logger,
        # logger_iter_interval=args.logger_iter_interval,
        # ckpt_save_time_interval=args.ckpt_save_time_interval,
        # use_logger_to_record=not args.use_tqdm_to_record,
        # show_gpu_stat=not args.wo_gpu_stat,
        # use_amp=args.use_amp,
        loss_fn=loss_fn,
        cfg=cfg,
    )
    print("finish")


if __name__ == "__main__":
    main()
