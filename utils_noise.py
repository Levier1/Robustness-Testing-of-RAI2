import os
import sys
import re
import time
import numpy as np
from conf import settings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

# 这里只是把utils里关于数据集加载以及一些设置用到的函数单独拿出来并修改新的加载和存储路径 其它的一些框架调用还是沿用utils里面的函数
class SubTrainDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # TODO: unify the following line, in case study, the below line does not exist.
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):

        return len(self.data)


# methods for loading sub-datasets of CIFAR-10/100 and Tiny-ImageNet


def get_dataset_hyperparam(dataset):
    if dataset == 'cifar10':
        return settings.CIFAR10_EPOCH, settings.CIFAR10_MILESTONES
    if dataset == 'cifar100':
        return settings.CIFAR100_EPOCH, settings.CIFAR100_MILESTONES
    if dataset == 'tinyimagenet':
        return settings.TINYIMAGENET_EPOCH, settings.TINYIMAGENET_MILESTONES


def get_dataset_mean_std(dataset):
    if dataset == 'cifar10':
        return settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD
    if dataset == 'cifar100':
        return settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD
    if dataset == 'tinyimagenet':
        return settings.TINYIMAGENET_TRAIN_MEAN, settings.TINYIMAGENET_TRAIN_STD


def get_intersection_mean_std_dict(dataset_name):
    '''
    get normalization mean and std for each intersections 0.0, 0.1, ..., 0.9, 1.0. used for evaluation.
    '''
    mean_std_dict = {}
    for s in (np.arange(11) / 10):
        Set1, Set2 = pickle.load(
            open(os.path.join(settings.DATA_PATH_NOISE, f'gaussian_noise/{dataset_name.upper()}_intersect_{s}.pkl'), 'rb'))
        mean = tuple((Set2[0] / 255).mean(axis=(0, 1, 2)))
        std = tuple((Set2[0] / 255).std(axis=(0, 1, 2)))
        mean_std_dict['int{}'.format(s)] = (mean, std)
    mean_std_dict['vic'] = mean_std_dict['int1.0']
    return mean_std_dict


# 先用简单的配合Test_accuracy的测试
def get_intersection_mean_std(dataset_name, inter_propor):
    # 根据传入的单一交集比例 (inter_propor) 获取该比例下的数据集的均值和标准差。
    # 构建文件路径，使用传入的比例值
    file_path = os.path.join(settings.DATA_PATH_NOISE, f'gaussian_noise/{dataset_name.upper()}_intersect_{inter_propor}.pkl')
    # 检查文件是否存在，避免 FileNotFoundError
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # 加载对应比例的数据集
    Set1, Set2 = pickle.load(open(file_path, 'rb'))
    # 计算均值和标准差（数据归一化前的像素值除以 255）
    mean = tuple((Set2[0] / 255).mean(axis=(0, 1, 2)))
    std = tuple((Set2[0] / 255).std(axis=(0, 1, 2)))
    # 返回该比例对应的均值和标准差
    return mean, std


def get_subtraining_dataloader_cifar10_intersect(propor=0.5, batch_size=16, num_workers=8, shuffle=True, sub_idx=1):
    X_set, y_set = \
    pickle.load(open(os.path.join(settings.DATA_PATH_NOISE, f'gaussian_noise/CIFAR10_intersect_{propor}.pkl'), 'rb'))[sub_idx]
    mean = tuple((X_set / 255).mean(axis=(0, 1, 2)))
    std = tuple((X_set / 255).std(axis=(0, 1, 2)))

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader, mean, std


def get_subtraining_dataloader_cifar100_intersect(propor=0.5, batch_size=16, num_workers=8, shuffle=True, sub_idx=1):
    X_set, y_set = \
    pickle.load(open(os.path.join(settings.DATA_PATH_NOISE, f'gaussian_noise/CIFAR100_intersect_{propor}.pkl'), 'rb'))[sub_idx]
    mean = tuple((X_set / 255).mean(axis=(0, 1, 2)))
    std = tuple((X_set / 255).std(axis=(0, 1, 2)))

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader, mean, std


def get_subtraining_dataloader_tinyimagenet_intersect(propor=0.5, batch_size=16, num_workers=8, shuffle=True,
                                                      sub_idx=1):
    X_set, y_set = \
    pickle.load(open(os.path.join(settings.DATA_PATH_NOISE, f'gaussian_noise/TINYIMAGENET_intersect_{propor}.pkl'), 'rb'))[
        sub_idx]
    mean = tuple((X_set / 255).mean(axis=(0, 1, 2)))
    std = tuple((X_set / 255).std(axis=(0, 1, 2)))

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tinyimagenet_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_training_loader, mean, std


def get_intersect_dataloader(dataset, propor, batch_size=16, num_workers=8, shuffle=True, sub_idx=1):
    if dataset == 'cifar10':
        return get_subtraining_dataloader_cifar10_intersect(propor=propor, batch_size=batch_size,
                                                            num_workers=num_workers, shuffle=shuffle, sub_idx=sub_idx)
    elif dataset == 'cifar100':
        return get_subtraining_dataloader_cifar100_intersect(propor=propor, batch_size=batch_size,
                                                             num_workers=num_workers, shuffle=shuffle, sub_idx=sub_idx)
    elif dataset == 'tinyimagenet':
        return get_subtraining_dataloader_tinyimagenet_intersect(propor=propor, batch_size=batch_size,
                                                                 num_workers=num_workers, shuffle=shuffle,
                                                                 sub_idx=sub_idx)


def get_training_dataloader_cifar10(mean, std, batch_size=16, num_workers=8, shuffle=True):
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_training = torchvision.datasets.CIFAR10(root=os.path.join(settings.DATA_PATH, 'CIFAR10'), train=True,
                                                    download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader


def get_training_dataloader_cifar100(mean, std, batch_size=16, num_workers=8, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root=os.path.join(settings.DATA_PATH, 'CIFAR100'), train=True,
                                                      download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_training_dataloader_tinyimagenet(mean, std, batch_size=16, num_workers=8, shuffle=True):
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tinyimagenet_training = datasets.ImageFolder(os.path.join(settings.DATA_PATH, 'tiny-imagenet-200/train/'),
                                                 transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_training_loader


def get_training_dataloader(dataset, mean, std, batch_size=16, num_workers=8, shuffle=True):
    if dataset == 'cifar10':
        return get_training_dataloader_cifar10(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers,
                                               shuffle=shuffle)
    if dataset == 'cifar100':
        return get_training_dataloader_cifar100(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers,
                                                shuffle=shuffle)
    if dataset == 'cifar10':
        return get_training_dataloader_tinyimagenet(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers,
                                                    shuffle=shuffle)


def get_test_dataloader_cifar10(mean, std, batch_size=16, num_workers=8, shuffle=False, pin_memory=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root=os.path.join(settings.DATA_PATH, 'CIFAR10'), train=False,
                                                download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)

    return cifar10_test_loader


def get_test_dataloader_cifar100(mean, std, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root=os.path.join(settings.DATA_PATH, 'CIFAR100'), train=False,
                                                  download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

    return cifar100_test_loader


def get_test_dataloader_tinyimagenet(mean, std, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    X_set, y_set = pickle.load(open(os.path.join(settings.DATA_PATH, 'similarity', 'TinyImagenet_test.pkl'), 'rb'))
    tinyimagenet_test = SubTrainDataset(X_set, list(y_set), transform=transform_test)
    tinyimagenet_test_loader = DataLoader(
        tinyimagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return tinyimagenet_test_loader


def get_test_dataloader(dataset, mean, std, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    if dataset == 'cifar10':
        return get_test_dataloader_cifar10(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers,
                                           shuffle=shuffle, pin_memory=pin_memory)
    elif dataset == 'cifar100':
        return get_test_dataloader_cifar100(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers,
                                            shuffle=shuffle, pin_memory=pin_memory)
    if dataset == 'tinyimagenet':
        return get_test_dataloader_tinyimagenet(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers,
                                                shuffle=shuffle, pin_memory=pin_memory)

