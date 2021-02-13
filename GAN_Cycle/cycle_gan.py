import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import warnings


def get_data_loader(image_type, image_dir='summer2winter_yosemite', image_size=128, batch_size=16, num_workers=0):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    image_path = image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))
    # Define Data Sets
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    # Data Loaders
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_data_loader, test_data_loader


dataloader_X, test_dataloader_X = get_data_loader(image_type='summer')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter')


def show_images(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(dataloader_X)
images, _ = dataiter.next()

fig = plt.figure(figsize=(12, 8))
show_images(torchvision.utils.make_grid(images))
