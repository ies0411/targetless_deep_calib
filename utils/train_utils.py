# from scipy.spatial.transform import Rotation as R
import mathutils
import torch
import torch.nn.functional as F
import tqdm

from .utils import lidar_project_depth, rotate_back


def train_model(
    model,
    optimizer,
    train_loader,
    lr_scheduler,
    start_epoch,
    total_epochs,
    start_iter,
    ckpt_save_dir,
    loss_fn,
    cfg=None,
):
    accumulated_iter = start_iter

    # use for disable data augmentation hook
    # hook_config = cfg.get("HOOK", None)
    # augment_disable_flag = False

    with tqdm.trange(
        start_epoch,
        total_epochs,
        desc="epochs",
        dynamic_ncols=True,
    ) as tbar:
        total_it_each_epoch = len(train_loader)
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            # if train_sampler is not None:
            #     train_sampler.set_epoch(cur_epoch)

            # train one epoch
            # if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
            #     cur_scheduler = lr_warmup_scheduler
            # else:
            #
            cur_scheduler = lr_scheduler
            ckpt_save_cnt = 1
            start_it = accumulated_iter % total_it_each_epoch
            local_loss = 0.0
            for cur_it in range(start_it, total_it_each_epoch):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(train_loader)
                    batch = next(dataloader_iter)
                    print("new iters")
                # print(f"batch : {batch}")
                lidar_input = []
                rgb_input = []
                lidar_gt = []
                shape_pad_input = []
                real_shape_input = []
                pc_rotated_input = []

                # gt pose
                batch["tr_error"] = batch["tr_error"].cuda()
                batch["rot_error"] = batch["rot_error"].cuda()
                for idx in range(len(batch["rgb"])):
                    # ProjectPointCloud in RT-pose
                    real_shape = [
                        batch["rgb"][idx].shape[1],
                        batch["rgb"][idx].shape[2],
                        batch["rgb"][idx].shape[0],
                    ]

                    batch["point_cloud"][idx] = batch["point_cloud"][
                        idx
                    ].cuda()  # pointcloud pose in camera plane
                    pc_lidar = batch["point_cloud"][idx].clone()

                    if cfg.MODEL.MAX_DEPTH < 80.0:
                        pc_lidar = pc_lidar[
                            :, pc_lidar[0, :] < cfg.MODEL.MAX_DEPTH
                        ].clone()

                    depth_gt, uv = lidar_project_depth(
                        pc_lidar, batch["calib"][idx], real_shape
                    )  # image_shape
                    depth_gt /= cfg.MODEL.MAX_DEPTH
                    R = mathutils.Quaternion(batch["rot_error"][idx]).to_matrix()
                    R.resize_4x4()
                    T = mathutils.Matrix.Translation(batch["tr_error"][idx])
                    # RT = T * R
                    RT = T @ R

                    pc_rotated = rotate_back(
                        batch["point_cloud"][idx], RT
                    )  # Pc` = RT * Pc

                    if cfg.MODEL.MAX_DEPTH < 80.0:
                        pc_rotated = pc_rotated[
                            :, pc_rotated[0, :] < cfg.MODEL.MAX_DEPTH
                        ].clone()

                    depth_img, uv = lidar_project_depth(
                        pc_rotated, batch["calib"][idx], real_shape
                    )  # image_shape
                    depth_img /= cfg.MODEL.MAX_DEPTH

                    # PAD ONLY ON RIGHT AND BOTTOM SIDE
                    rgb = batch["rgb"][idx].cuda()
                    shape_pad = [0, 0, 0, 0]

                    shape_pad[3] = cfg.DATA_CONFIG.IMAGE_SHAPE[0] - rgb.shape[1]  # // 2
                    shape_pad[1] = (
                        cfg.DATA_CONFIG.IMAGE_SHAPE[1] - rgb.shape[2]
                    )  # // 2 + 1

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
                # rgb_show = rgb_input.clone()
                # lidar_show = lidar_input.clone()
                rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
                lidar_input = F.interpolate(
                    lidar_input, size=[256, 512], mode="bilinear"
                )
                # end_preprocess = time.time()
                ##train
                model.train()
                optimizer.zero_grad()

                # Run model
                transl_err, rot_err = model(rgb_input, lidar_input)
                print(f"transl_err : {transl_err}")
                print(f"rot_err : {rot_err}")
                # if loss == "points_distance" or loss == "combined":
                losses = loss_fn(
                    batch["point_cloud"],
                    batch["tr_error"],
                    batch["rot_error"],
                    transl_err,
                    rot_err,
                )

                losses["total_loss"].backward()
                print("backward?")
                optimizer.step()
                # loss, R_predicted, T_predicted = train(
                #     model,
                #     optimizer,
                #     rgb_input,
                #     lidar_input,
                #     sample["tr_error"],
                #     sample["rot_error"],
                #     loss_fn,
                #     sample["point_cloud"],
                #     config["loss"],
                # )
                for key in losses.keys():
                    if losses[key].item() != losses[key].item():
                        raise ValueError("Loss {} is NaN".format(key))

                local_loss += losses["total_loss"].item()
                # trained_epoch = cur_epoch + 1
                # if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                #     ckpt_list = glob.glob(str(ckpt_save_dir / "checkpoint_epoch_*.pth"))
                #     ckpt_list.sort(key=os.path.getmtime)

                #     if ckpt_list.__len__() >= max_ckpt_save_num:
                #         for cur_file_idx in range(
                #             0, len(ckpt_list) - max_ckpt_save_num + 1
                #         ):
                #             os.remove(ckpt_list[cur_file_idx])

                #     ckpt_name = ckpt_save_dir / ("checkpoint_epoch_%d" % trained_epoch)
                #     save_checkpoint(
                #         checkpoint_state(
                #             model, optimizer, trained_epoch, accumulated_iter
                #         ),
                #         filename=ckpt_name,
                #     )
