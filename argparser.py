import os
import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="background penalty, should be less then 1"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=2,
        help="How often to save checkpoints (epochs)",
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        default=10,
        help="How often to perform validation (epochs)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of images in each batch"
    )
    parser.add_argument(
        "--context_path",
        type=str,
        default="resnet18",
        help="The context path model you are using, resnet18,resnet50, resnet101.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate used for train"
    )
    parser.add_argument(
        "--data_root", type=str, default="/content/root_drive/MyDrive/data", help="path of training data"
    )
    parser.add_argument(
        "--trained", type=str, default="15-5-ov", help="training steps"
    )
    parser.add_argument("--num_workers", type=int,
                        default=4, help="num of workers")
    parser.add_argument(
        "--num_classes", type=int, default=6, help="num of object classes (with void)"
    )
    parser.add_argument(
        "--cuda", type=str, default="0", help="GPU ids used for training"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="whether to user gpu for training"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--save_model_dir", type=str, default="checkpoints", help="path to save dir"
    )
    parser.add_argument(
        "--save_model_path", type=str, default="checkpoints", help="path to save model"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer, support rmsprop, sgd, adam",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="crossentropy",
        help="loss function, dice or crossentropy",
    )
    parser.add_argument(
        "--use_amp",
        type=bool,
        default=True,
        help="use automatic mixed precision",
    )
    parser.add_argument(
        "--use_lrScheduler",
        type=bool,
        default=False,
        help="Permit the use of lr Scheduler",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=310,
        help="Random crop size for data augmentation during training",
    )
    parser.add_argument(
        "--further_data_aug",
        type=bool,
        default=False,
        help="Permit use of new data augmentations for training",
    )
    # Incremental parameters
    parser.add_argument("--task", type=str, default="15-5-ov", choices=["offline-ov","15-5-ov","15-5s-ov"],
                        help="Task to be executed (default: 15-5-ov)")
    parser.add_argument("--step", type=int, default="0",help="incremental step")
    parser.add_argument("--overlap", action='store_true', default=False,
                        help="Use this to not use the new classes in the old training set")
    # Datset Options (dataset_directory , dataset_name)
    parser.add_argument("--dataset", type=str, default='voc',choices=['voc'], help='Name of dataset')
    # Validation Options
    parser.add_argument("--val_on_trainset", action='store_true', default=False, help="enable validation on train set (default: False)")
    parser.add_argument("--cross_val", action='store_true', default=False, help="If validate on training or on validation (default: Train)")
    parser.add_argument("--crop_val", action='store_false', default=True, help='do crop for validation (default: True)')

    return parser
