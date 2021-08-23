import os
import random
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np
from .subset import Subset

from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = image_set
        base_dir = "PascalVOC12"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')


        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt') # the split file address

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(
            voc_root, x[1][1:])) for x in file_names]
        # images =  'path/JPEGImages/2007_000032.jpg' , 'path/SegmentationClassAug/2007_000032.png'
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationIncremental(data.Dataset):
    def __init__(self, root, train=True, transform=None, labels=None,
                 idxs_path=None):

        voc = VOCSegmentation(root, 'train_aug' if train else 'trainval_aug', transform=None)
        idxs = None
        self.labels = labels

        if labels is not None:
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                raise NotImplementedError
            if train:
                masking_value = 0
            else:
                masking_value = 255
            z = np.arange(len(labels))
          
            self.inverted_order = dict(zip(labels, z))
            self.inverted_order[255] = masking_value
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value))

            self.dataset = Subset(voc, idxs, transform, target_transform)

        else:
            self.dataset = voc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
