import dataclasses

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

@dataclasses.dataclass
class DatasetInfo:
    dataset_name    : str
    num_classes     : int
    num_channels    : int
    image_size      : tuple
    torch_class     : type

DATASET_INFO = {
    "CIFAR10"       : DatasetInfo("CIFAR10",        10,  3, (32, 32), datasets.CIFAR10),
    "CIFAR100"      : DatasetInfo("CIFAR100",       100, 3, (32, 32), datasets.CIFAR100),
    "MNIST"         : DatasetInfo("MNIST",          10,  1, (28, 28), datasets.MNIST),
    "FashionMNIST"  : DatasetInfo("FashionMNIST",   10,  1, (28, 28), datasets.FashionMNIST),
    "KMNIST"        : DatasetInfo("KMNIST",         10,  1, (28, 28), datasets.KMNIST),
    "SVHN"          : DatasetInfo("SVHN",           10,  3, (32, 32), datasets.SVHN),
    "STL10"         : DatasetInfo("STL10",          10,  3, (96, 96), datasets.STL10),
}

# Calculate MEAN and STDDEV values on the dataset
def calculate_mean_std(dataset_info):
    # Assign the data to a new dataset object - each image becomes a tensor through transforms.ToTensor()

    if  dataset_info.torch_class == datasets.SVHN:                           # NOTE: SVHN uses split='train' as opposed to train = true/false
        dataset = dataset_info.torch_class(root = './datasets', split = 'train', transform = transforms.ToTensor(), download = True)

    else:
        dataset = dataset_info.torch_class(root = './datasets', train = True, transform = transforms.ToTensor(), download = True)

    # Stack each image in the dataset into a final tensor (N, C, W, H)
    data = torch.stack([img for img, _label in dataset])  #   _label is dropped, N (imageNumber) added.

    (std, mean) = torch.std_mean(data, dim = (0, 2, 3), keepdim = False)
    return (mean, std)


def create_data_loaders(train_dataset, val_dataset, test_dataset, dataloading_config):

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = dataloading_config["batch_size"], shuffle = dataloading_config["shuffle_train_data"],  num_workers = dataloading_config["num_workers"])
    val_loader   = DataLoader(val_dataset,   batch_size = dataloading_config["batch_size"], shuffle = False, num_workers = dataloading_config["num_workers"])
    test_loader  = DataLoader(test_dataset,  batch_size = dataloading_config["batch_size"], shuffle = False, num_workers = dataloading_config["num_workers"])

    # Package loaders
    train_val_test_dataLoaders = (train_loader,val_loader,test_loader)
    return train_val_test_dataLoaders

def make_validation_splits(train_dataset, dataset_config):

    val_size   = int(len(train_dataset) * dataset_config["val_split"])
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset

def print_dataset_info(dataset_info):

    print(f"Dataset     :  {dataset_info.dataset_name}")
    print(f"Classes     :  {dataset_info.num_classes}")
    print(f"Channels    :  {dataset_info.num_channels}")
    print(f"Image Size  :  {dataset_info.image_size}")
