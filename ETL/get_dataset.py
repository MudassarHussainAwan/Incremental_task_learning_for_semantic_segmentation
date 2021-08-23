import os
import tasks
import torch
from ETL import transform
from ETL import VOCSegmentationIncremental



def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = transform.Compose([
            transform.Resize(size=opts.crop_size),
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    else:
        raise NotImplementedError
    labels,labels_cum, path_base = tasks.get_task_labels(
        opts.dataset, name=opts.task, step= opts.step)

    train_dst = dataset(root=opts.data_root, train=True, transform=train_transform,
                        labels=labels,
                        idxs_path=path_base + f"/train-{opts.step}.npy")

    if opts.cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst)-train_len
        train_dst, val_dst = torch.utils.data.random_split(
            train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(root=opts.data_root, train=False, transform=val_transform,
                          labels=labels,
                          idxs_path=path_base + f"/val-{opts.step}.npy")

    test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
                       labels=labels_cum,
                       idxs_path=path_base + f"/test_on_val-{opts.step}.npy")

    return (train_dst, val_dst, test_dst, len(labels))
