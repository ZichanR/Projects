from torch.utils.data import Subset
from PIL import Image
from torch.utils.data import Dataset
from base.base_dataset import BaseADDataset
from torch.utils.data import DataLoader
from .preprocessing import create_semisupervised_setting

import os
import pickle
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
import warnings
import json
import datetime as dt

class VisAnaML_Dataset(BaseADDataset):

    def __init__(self, root: str, args, ratio_unlabel: float = 0.0, n_pollution: int = 0):
        super().__init__(root)
        
        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = 0
        self.outlier_classes = tuple([1])

        # # MNIST preprocessing: feature scaling to [0, 1]
        # transform = transforms.ToTensor()
        # target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        self.train_set = MISLABELVisAnaML(root=self.root, args=args, train=True, traintestsplit = args.traintestsplit)

        # # Create semi-supervised setting
        # idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
        #                                                      self.outlier_classes, ratio_unlabel, n_pollution)
        # train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # # Subset train_set to semi-supervised setup
        # self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = VisAnaML(root=self.root, train=False,traintestsplit = args.traintestsplit)
        
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader

class VisAnaML(Dataset):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, root: str, train=True, random_state=None, load_save=True, traintestsplit = 'date'):
        super(Dataset, self).__init__()
        
        self.root = '%s/VisAnaML'%(root)
        self.train = train  # training set or test set
        self.traintestsplit = traintestsplit
        
        if self.train:
            if self.traintestsplit == 'random':
                self.data_file = '%s/random_train_data.dat' % (self.root)
                self.label_file = '%s/random_train_label.json' % (self.root)
                self.key_file = '%s/random_train_keydt.info' % (self.root)
            if self.traintestsplit == 'date':
                self.data_file = '%s/date_train_data.dat' % (self.root)
                self.label_file = '%s/date_train_label.json' % (self.root)
                self.key_file = '%s/date_train_keydt.info' % (self.root)
        else:
            if self.traintestsplit == 'random':
                self.data_file = '%s/random_test_data.dat' % (self.root)
                self.label_file = '%s/random_test_label.json' % (self.root)
                self.key_file = '%s/random_test_keydt.info' % (self.root)
            if self.traintestsplit == 'date':
                self.data_file = '%s/date_test_data.dat' % (self.root)
                self.label_file = '%s/date_test_label.json' % (self.root)
                self.key_file = '%s/date_test_keydt.info' % (self.root)
            
        if load_save:
            self.download()
        
        data = np.fromfile(self.data_file)
        data = data.reshape(-1,96,21)
        with open(self.label_file, 'r') as f:
            label = [int(line.rstrip('\n')) for line in f]
        with open(self.key_file, 'r') as f:
            keys = json.load(f)
        
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.targets = torch.tensor(np.array(label), dtype=torch.int64)
        self.semi_targets = torch.zeros_like(self.targets).float()
        self.keys = np.array(keys)
        
    def _check_exists(self):
        return os.path.exists(self.data_file) and os.path.exists(self.label_file)

    def download(self):
        """Download the ODDS dataset if it doesn't exist in root already."""

        if self._check_exists():
            return

        # download file
        # download_url(self.urls[self.dataset_name], self.root, self.file_name)
        label_filename = '%s/listlabel.json'%(self.root)
        data_filename = '%s/transformedlist.json'%(self.root)

        data = []
        keys = pd.DataFrame(columns =['KEY','Time'])
        with open(data_filename, "r") as f:
            load_data = json.loads(f.read())
        for json_key_date_data in load_data:
            df = pd.DataFrame.from_dict(json_key_date_data)
            keys = keys.append(pd.DataFrame(data={'KEY':df['KEY'][0],'Time':df['Time'][0].split(' ')[0]},index=[0]), ignore_index=True)
            df.drop(columns=['KEY','Time'],inplace=True)
            data.append(df.values)
            
        with open(label_filename, 'r') as f:
            label = [int(line.rstrip('\n')) for line in f]
        
        if self.traintestsplit == 'date':
            dates = keys['Time'].unique()
            dates = [dt.datetime.strptime(dates[i], '%Y-%m-%d') for i in range(len(dates))]
            dates.sort()
            train_dates = [d.strftime('%Y-%m-%d') for d in dates[:int(len(dates)*0.8)]]
            test_dates = [d.strftime('%Y-%m-%d') for d in dates[int(len(dates)*0.8):]]
            train_idx = keys.index[keys.isin({'Time': train_dates})['Time']].tolist()
            test_idx = keys.index[keys.isin({'Time': test_dates})['Time']].tolist()
            train_data_filename = '%s/date_train_data.dat' % (self.root)
            test_data_filename = '%s/date_test_data.dat' % (self.root)
            train_label_filename = '%s/date_train_label.json' % (self.root)
            test_label_filename = '%s/date_test_label.json' % (self.root)
            train_key_filename = '%s/date_train_keydt.info' % (self.root)
            test_key_filename = '%s/date_test_keydt.info' % (self.root)
            
        if self.traintestsplit == 'random':
            label = pd.Series(label, name = 'label')
            index_0 = label.index[label == 0].tolist()
            index_1 = label.index[label == 1].tolist()
            random.shuffle(index_0)
            random.shuffle(index_1)
            train_idx = index_0[:int(len(index_0)*0.8)] + index_1[:int(len(index_1)*0.8)]
            test_idx = index_0[int(len(index_0)*0.8):] + index_1[int(len(index_1)*0.8):]
            train_data_filename = '%s/random_train_data.dat' % (self.root)
            test_data_filename = '%s/random_test_data.dat' % (self.root)
            train_label_filename = '%s/random_train_label.json' % (self.root)
            test_label_filename = '%s/random_test_label.json' % (self.root)
            train_key_filename = '%s/random_train_keydt.info' % (self.root)
            test_key_filename = '%s/random_test_keydt.info' % (self.root)
        
        train_data = [data[idx] for idx in train_idx]
        train_label = [label[idx] for idx in train_idx]
        train_key = [keys.iloc[idx].tolist() for idx in train_idx]
        test_data = [data[idx] for idx in test_idx]
        test_label = [label[idx] for idx in test_idx]
        test_key = [keys.iloc[idx].tolist() for idx in test_idx]
        
        train_data = np.array(train_data)
        train_data.tofile(train_data_filename)
        
        test_data = np.array(test_data)
        test_data.tofile(test_data_filename)
        
        with open(train_label_filename, 'w') as f:
            for l in train_label:
                f.write(str(l) + '\n')
            
        with open(test_label_filename, 'w') as f:
            for l in test_label:
                f.write(str(l) + '\n')

        with open(train_key_filename, "w") as f:
            json.dump(train_key, f)
            
        with open(test_key_filename, "w") as f:
            json.dump(test_key, f)
        
        print('Done!')
        
    def __len__(self):
        return len(self.data)
    
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
        # img = Image.fromarray(img.numpy(), mode='L')

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, semi_target, index
    
    def findfuture(self, index):
        keys = self.keys[index]
        fut_keys = [[str(k[0]),(dt.datetime.strptime(k[1], '%Y-%m-%d')+dt.timedelta(days=1)).strftime('%Y-%m-%d')] for k in keys]
        fut_index = []
        exists = []
        for k in fut_keys:
            a = np.argwhere(np.isin(self.keys[:,0], k[0])).flatten().ravel()
            b = np.argwhere(np.isin(self.keys[a,1], k[1])).flatten().ravel()
            if bool(b):
                exists.append(True)
                fut_index.append(a[b].ravel()[0])
            else:
                exists.append(False)
        
        return self.data[fut_index], exists
    
class MISLABELVisAnaML(VisAnaML):
    
    def __init__(self, root, args, normal_classes=0, outlier_classes=1, train=True,  traintestsplit = 'date'):
        super(MISLABELVisAnaML, self).__init__(root, train, traintestsplit=traintestsplit)
        if self.train:
            np.random.seed(args.rand_number)
            # Create semi-supervised setting
            idx, _, semi_targets = create_semisupervised_setting(self.targets.cpu().data.numpy(), normal_classes,
                                                                 outlier_classes, args.ratio_unlabel, args.n_pollution)
            self.semi_targets[idx] = torch.tensor(semi_targets).float()  # set respective semi-supervised labels
            self.gen_mislabeled_data(mislabel_type=args.mislabel_type, mislabel_ratio=args.mislabel_ratio)
        
    def gen_mislabeled_data(self, mislabel_type, mislabel_ratio):
        """Gen a list of imbalanced training data, and replace the origin data with the generated ones."""
        new_semi_targets = []
        new_targets = []
        num_cls = 2

        if mislabel_type == 'agnostic':
            for i, semi_target in enumerate(self.semi_targets):
                if semi_target == 0:
                    new_semi_targets.append(int(semi_target))
                    new_targets.append(-1)
                else:
                    if np.random.rand() < mislabel_ratio:
                        new_semi_target = int(semi_target)
                        while new_semi_target == int(semi_target):
                            new_target = np.random.randint(num_cls)
                            new_semi_target = int(new_target*-2+1)
                        new_semi_targets.append(int(new_semi_target))
                        new_targets.append(int(new_target))
                    else:
                        new_semi_targets.append(int(semi_target))
                        new_targets.append(int(semi_target/-2+0.5))
        elif mislabel_type == 'asym_o_n': # outlier to normal
            # ordered_list = [-1,1] # outlier
            permu_list = [1,0]
            for i, semi_target in enumerate(self.semi_targets):
                if np.random.rand() < mislabel_ratio:
                    new_semi_target = permu_list[0]
                    new_target = permu_list[1]
                    new_semi_targets.append(new_semi_target)
                    new_targets.append(new_target)
                else:
                    new_semi_targets.append(int(semi_target))
                    new_targets.append(int(semi_target/-2+0.5))
        elif mislabel_type == 'asym_n_o': # normal to outlier  
            # ordered_list = [1,0] # normal
            permu_list = [-1,1]
            for i, semi_target in enumerate(self.semi_targets):
                if np.random.rand() < mislabel_ratio:
                    new_semi_target = permu_list[0]
                    new_target = permu_list[1]
                    new_semi_targets.append(new_semi_target)
                    new_targets.append(new_target)
                else:
                    new_semi_targets.append(semi_target)
                    new_targets.append(int(semi_target/-2+0.5))
        else:
            warnings.warn('Noise type is not listed')
        
        self.real_targets = self.targets
        self.semi_targets = torch.Tensor(new_semi_targets).int()
        self.targets = torch.Tensor(new_targets).int()


    def __getitem__(self, index):      
        if self.train:
            img, target, real_target, semi_target = self.data[index], int(self.targets[index]), int(self.real_targets[index]), int(self.semi_targets[index])
        else:
            img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        
        if self.train:
            return img, target, semi_target, real_target, index
        else:
            return img, target, semi_target, index