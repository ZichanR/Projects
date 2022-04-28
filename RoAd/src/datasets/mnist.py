from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random
import numpy as np
import warnings


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, args, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MISLABELMNIST(root=self.root, args=args, train=True, download=True,
                            transform=transform, target_transform=target_transform)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.indices = idx
        self.train_set = Subset(train_set, self.indices)

        # Get test set
        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                download=True)
        
    def train_set_reload(self, coresets):
        self.train_set = Subset(self.train_set.dataset, coresets)


class MyMNIST(MNIST):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index

class MISLABELMNIST(MyMNIST):
    
    def __init__(self, root, args, train=True, transform=None, target_transform=None, download=False):
        super(MISLABELMNIST, self).__init__(root, train, transform, target_transform, download)
        if self.train:
            np.random.seed(args.rand_number)
            self.gen_mislabeled_data(mislabel_type=args.mislabel_type, mislabel_ratio=args.mislabel_ratio)
        
    def gen_mislabeled_data(self, mislabel_type, mislabel_ratio):
        """Gen a list of imbalanced training data, and replace the origin data with the generated ones."""
        new_targets = []
        num_cls = np.max(self.targets.numpy()) + 1
            
        if mislabel_type == 'agnostic':
            for i, target in enumerate(self.targets):
                if np.random.rand() < mislabel_ratio:
                    new_target = int(target)
                    while new_target == int(target):
                        new_target = np.random.randint(num_cls)
                    new_targets.append(new_target)
                else:
                    new_targets.append(int(target))
        elif mislabel_type == 'asym':
            ordered_list = np.arange(num_cls)
            while True: 
                permu_list = np.random.permutation(num_cls)
                if np.any(ordered_list == permu_list):
                    continue
                else:
                    break
            for i, target in enumerate(self.targets):
                if np.random.rand() < mislabel_ratio:
                    new_target = permu_list[target]
                    new_targets.append(new_target)
                else:
                    new_targets.append(target)
        else:
            warnings.warn('Noise type is not listed')
        
        self.real_targets = self.targets
        self.targets = torch.Tensor(new_targets).int()

    def estimate_label_acc(self, indices):
        targets_np = np.array(self.targets)[indices]
        real_targets_np = np.array(self.real_targets)[indices]
        label_acc = np.sum((targets_np == real_targets_np)) / len(targets_np)
        return label_acc

    def fetch(self, targets, indices):
        whole_targets_np = np.array(self.targets)[indices]
        idx_dict = {}
        
        idx_dict[0] = np.where(whole_targets_np == 0)[0]
        idx_dict[1] = np.where(whole_targets_np != 0)[0]

        idx_list = []
        for target in targets:
            idx_list.append(np.random.choice(idx_dict[target.item()], 1))
        idx_list = np.array(idx_list).flatten()
        imgs = []
        for idx in idx_list:
            img = self.data[indices][idx].numpy()
            img = Image.fromarray(img)
            img = self.transform(img)
            imgs.append(img[None, ...])
        train_data = torch.cat(imgs, dim=0)
        return train_data

    def __getitem__(self, index):      
        if self.train:
            img, target, real_target, semi_target = self.data[index], int(self.targets[index]), int(self.real_targets[index]), int(self.semi_targets[index])
        else:
            img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.train:
            return img, target, semi_target, real_target, index
        else:
            return img, target, semi_target, index