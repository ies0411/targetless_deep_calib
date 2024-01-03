from functools import partial

import torch
from kitti_odom import KittiOdomDataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from utils.utils import worker_init_fn

__all__ = {
    "KittiOdom": KittiOdomDataset,
}


def build_dataloader(
    dataset_cfg,
    # class_names,
    batch_size,
    # dist,
    root_path=None,
    workers=4,
    seed=None,
    # logger=None,
    training=True,
    # total_epochs=0,
):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        # class_names=class_names,
        # root_path=root_path,
        training=training,
        # logger=logger,
    )

    # if dist:
    #     if training:
    #         sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     else:
    #         rank, world_size = common_utils.get_dist_info()
    #         sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    # else:
    sampler = None
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
        shuffle=(sampler is None) and training,
        collate_fn=dataset.merge_inputs,
        drop_last=False,
        sampler=sampler,
        timeout=0,
        worker_init_fn=partial(worker_init_fn, seed=seed),
    )

    return dataset, dataloader, sampler


# def build_dataset():
