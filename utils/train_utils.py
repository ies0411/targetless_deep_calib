import torch
import torch.nn.functional as F
import tqdm


def train_model(
    model,
    optimizer,
    train_loader,
    model_func,
    lr_scheduler,
    optim_cfg,
    start_epoch,
    total_epochs,
    start_iter,
    rank,
    tb_log,
    ckpt_save_dir,
    train_sampler=None,
    lr_warmup_scheduler=None,
    ckpt_save_interval=1,
    max_ckpt_save_num=50,
    merge_all_iters_to_one_epoch=False,
    use_amp=False,
    use_logger_to_record=False,
    logger=None,
    logger_iter_interval=None,
    ckpt_save_time_interval=None,
    show_gpu_stat=False,
    cfg=None,
):
    accumulated_iter = start_iter

    # use for disable data augmentation hook
    # hook_config = cfg.get("HOOK", None)
    # augment_disable_flag = False

    with tqdm.trange(
        start_epoch, total_epochs, desc="epochs", dynamic_ncols=True, leave=(rank == 0)
    ) as tbar:
        total_it_each_epoch = len(train_loader)
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            # if train_sampler is not None:
            #     train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            ckpt_save_cnt = 1
            start_it = accumulated_iter % total_it_each_epoch
            for cur_it in range(start_it, total_it_each_epoch):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(train_loader)
                    batch = next(dataloader_iter)
                    print("new iters")

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

                    if config["max_depth"] < 80.0:
                        pc_lidar = pc_lidar[
                            :, pc_lidar[0, :] < config["max_depth"]
                        ].clone()

                    depth_gt, uv = lidar_project_depth(
                        pc_lidar, batch["calib"][idx], real_shape
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
                lidar_input = F.interpolate(
                    lidar_input, size=[256, 512], mode="bilinear"
                )
                # end_preprocess = time.time()
                ##train
                model.train()

                optimizer.zero_grad()

                # Run model
                transl_err, rot_err = model(rgb_img, refl_img)

                if loss == "points_distance" or loss == "combined":
                    losses = loss_fn(
                        point_clouds, target_transl, target_rot, transl_err, rot_err
                    )
                else:
                    losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

                losses["total_loss"].backward()
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
                for key in loss.keys():
                    if loss[key].item() != loss[key].item():
                        raise ValueError("Loss {} is NaN".format(key))

                trained_epoch = cur_epoch + 1
                if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                    ckpt_list = glob.glob(str(ckpt_save_dir / "checkpoint_epoch_*.pth"))
                    ckpt_list.sort(key=os.path.getmtime)

                    if ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(
                            0, len(ckpt_list) - max_ckpt_save_num + 1
                        ):
                            os.remove(ckpt_list[cur_file_idx])

                    ckpt_name = ckpt_save_dir / ("checkpoint_epoch_%d" % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(
                            model, optimizer, trained_epoch, accumulated_iter
                        ),
                        filename=ckpt_name,
                    )


# for epoch in range(starting_epoch, config["epochs"] + 1):
#     # EPOCH = epoch
#     print("This is %d-th epoch" % epoch)
#     epoch_start_time = time.time()
#     total_train_loss = 0
#     local_loss = 0.0
#     scheduler.step(epoch%100)

# for batch_idx, sample in enumerate(TrainImgLoader):
#             # print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
#             start_time = time.time()
#             lidar_input = []
#             rgb_input = []
#             lidar_gt = []
#             shape_pad_input = []
#             real_shape_input = []
#             pc_rotated_input = []

#             # gt pose
#             sample["tr_error"] = sample["tr_error"].cuda()
#             sample["rot_error"] = sample["rot_error"].cuda()

#             start_preprocess = time.time()
#             for idx in range(len(sample["rgb"])):
#                 # ProjectPointCloud in RT-pose
#                 real_shape = [
#                     sample["rgb"][idx].shape[1],
#                     sample["rgb"][idx].shape[2],
#                     sample["rgb"][idx].shape[0],
#                 ]

#                 sample["point_cloud"][idx] = sample["point_cloud"][
#                     idx
#                 ].cuda()  # pointcloud pose in camera plane
#                 pc_lidar = sample["point_cloud"][idx].clone()

#                 if config["max_depth"] < 80.0:
#                     pc_lidar = pc_lidar[:, pc_lidar[0, :] < config["max_depth"]].clone()

#                 depth_gt, uv = lidar_project_depth(
#                     pc_lidar, sample["calib"][idx], real_shape
#                 )  # image_shape
#                 depth_gt /= config["max_depth"]

#                 R = mathutils.Quaternion(sample["rot_error"][idx]).to_matrix()
#                 R.resize_4x4()
#                 T = mathutils.Matrix.Translation(sample["tr_error"][idx])
#                 RT = T * R

#                 pc_rotated = rotate_back(
#                     sample["point_cloud"][idx], RT
#                 )  # Pc` = RT * Pc

#                 if config["max_depth"] < 80.0:
#                     pc_rotated = pc_rotated[
#                         :, pc_rotated[0, :] < config["max_depth"]
#                     ].clone()

#                 depth_img, uv = lidar_project_depth(
#                     pc_rotated, sample["calib"][idx], real_shape
#                 )  # image_shape
#                 depth_img /= config["max_depth"]

#                 # PAD ONLY ON RIGHT AND BOTTOM SIDE
#                 rgb = sample["rgb"][idx].cuda()
#                 shape_pad = [0, 0, 0, 0]

#                 shape_pad[3] = img_shape[0] - rgb.shape[1]  # // 2
#                 shape_pad[1] = img_shape[1] - rgb.shape[2]  # // 2 + 1

#                 rgb = F.pad(rgb, shape_pad)
#                 depth_img = F.pad(depth_img, shape_pad)
#                 depth_gt = F.pad(depth_gt, shape_pad)

#                 rgb_input.append(rgb)
#                 lidar_input.append(depth_img)
#                 lidar_gt.append(depth_gt)
#                 real_shape_input.append(real_shape)
#                 shape_pad_input.append(shape_pad)
#                 pc_rotated_input.append(pc_rotated)

#             lidar_input = torch.stack(lidar_input)
#             rgb_input = torch.stack(rgb_input)
#             rgb_show = rgb_input.clone()
#             lidar_show = lidar_input.clone()
#             rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
#             lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
#             end_preprocess = time.time()
#             loss, R_predicted, T_predicted = train(
#                 model,
#                 optimizer,
#                 rgb_input,
#                 lidar_input,
#                 sample["tr_error"],
#                 sample["rot_error"],
#                 loss_fn,
#                 sample["point_cloud"],
#                 config["loss"],
#             )

#             for key in loss.keys():
#                 if loss[key].item() != loss[key].item():
#                     raise ValueError("Loss {} is NaN".format(key))

#             # if batch_idx % config["log_frequency"] == 0:
#             #     show_idx = 0
#             #     # output image: The overlay image of the input rgb image
#             #     # and the projected lidar pointcloud depth image
#             #     rotated_point_cloud = pc_rotated_input[show_idx]
#             #     R_predicted = quat2mat(R_predicted[show_idx])
#             #     T_predicted = tvector2mat(T_predicted[show_idx])
#             #     RT_predicted = torch.mm(T_predicted, R_predicted)
#             #     rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

#             #     depth_pred, uv = lidar_project_depth(
#             #         rotated_point_cloud,
#             #         sample["calib"][show_idx],
#             #         real_shape_input[show_idx],
#             #     )  # or image_shape
#             #     depth_pred /= config["max_depth"]
#             #     depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

#             #     pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
#             #     input_show = overlay_imgs(
#             #         rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0)
#             #     )
#             #     gt_show = overlay_imgs(
#             #         rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0)
#             #     )

#             #     pred_show = torch.from_numpy(pred_show)
#             #     pred_show = pred_show.permute(2, 0, 1)
#             #     input_show = torch.from_numpy(input_show)
#             #     input_show = input_show.permute(2, 0, 1)
#             #     gt_show = torch.from_numpy(gt_show)
#             #     gt_show = gt_show.permute(2, 0, 1)

#             local_loss += loss["total_loss"].item()

#             if batch_idx % 50 == 0 and batch_idx != 0:
#                 print(
#                     f"Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, "
#                     f"time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, "
#                     # f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
#                     f"time for 50 iter: {time.time()-time_for_50ep:.4f}"
#                 )
#                 time_for_50ep = time.time()
#                 # _run.log_scalar("Loss", local_loss / 50, train_iter)
#                 local_loss = 0.0
#             total_train_loss += loss["total_loss"].item() * len(sample["rgb"])
#             train_iter += 1
#             # total_iter += len(sample['rgb'])

#         print("------------------------------------")
#         print(
#             "epoch %d total training loss = %.3f"
#             % (epoch, total_train_loss / len(dataset_train))
#         )
#         print("Total epoch time = %.2f" % (time.time() - epoch_start_time))
#         print("------------------------------------")
