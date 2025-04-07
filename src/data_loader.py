import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.utils.data

def get_cifar10_loader(split='train', batch_size=128, workers=0, subset_size=0):
    """
    Args:
        split (str): 'train' or 'val'
        batch_size (int): Batch size for DataLoader
        workers (int): Number of worker threads
        subset_size (int): If > 0, return a subset of the dataset
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split == 'train':
        dataset = datasets.CIFAR10(
            root='./data', train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        shuffle = True
    elif split == 'val':
        dataset = datasets.CIFAR10(
            root='./data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        shuffle = False
    else:
        raise ValueError("split must be either 'train' or 'val'")

    if subset_size > 0:
        dataset = torch.utils.data.Subset(dataset, range(subset_size))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=workers,
        pin_memory=True)

    return loader

