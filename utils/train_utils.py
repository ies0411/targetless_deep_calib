def train(
    model,
    optimizer,
    rgb_img,
    refl_img,
    target_transl,
    target_rot,
    loss_fn,
    point_clouds,
    loss,
):
    model.train()

    optimizer.zero_grad()

    # Run model
    transl_err, rot_err = model(rgb_img, refl_img)

    if loss == "points_distance" or loss == "combined":
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    losses["total_loss"].backward()
    optimizer.step()

    return losses, rot_err, transl_err


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
    hook_config = cfg.get("HOOK", None)
    augment_disable_flag = False

    with tqdm.trange(
        start_epoch, total_epochs, desc="epochs", dynamic_ncols=True, leave=(rank == 0)
    ) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, "merge_all_iters_to_one_epoch")
            train_loader.dataset.merge_all_iters_to_one_epoch(
                merge=True, epochs=total_epochs
            )
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            augment_disable_flag = disable_augmentation_hook(
                hook_config,
                dataloader_iter,
                total_epochs,
                cur_epoch,
                cfg,
                augment_disable_flag,
                logger,
            )
            accumulated_iter = train_one_epoch(
                model,
                optimizer,
                train_loader,
                model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=optim_cfg,
                rank=rank,
                tbar=tbar,
                tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cur_epoch=cur_epoch,
                total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record,
                logger=logger,
                logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir,
                ckpt_save_time_interval=ckpt_save_time_interval,
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp,
            )

            # save trained model
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
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                    filename=ckpt_name,
                )
