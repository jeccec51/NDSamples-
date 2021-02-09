import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

transform = transforms.ToTensor()
# SVHN Training Data Sets
svhn_train = datasets.SVHN(root='data/', split='train', download=True, transform=transform)
batch_size = 128
num_workers = 0

# Build Data Loader
train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

# Visualize Data
dataiter = iter(train_loader)
images, labels = dataiter.next()

fig = plt.figure(figsize=(25, 4))
plot_size = 20

for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size / 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(str(labels[idx].item()))


def scale(x, feature_range=(-1, 1)):
    min1, max1 = feature_range
    x = x * (max1 - min1) + min1
    return x


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = out.view(-1, self.conv_dim * 4 * 4 * 4)
        out = self.fc(out)
        return out


def deconv(in_channels, out_channels, kernal_size, stride=2, padding=1, batch_norm=True):
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernal_size, stride, padding, bias=False)
    layers.append(transpose_conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.sequential(*layers)


class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.fc1 = nn.Linear(z_size, conv_dim * 4 * 4 * 4)
        self.t_conv1 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.t_conv2 = deconv(conv_dim * 2, conv_dim, 4)
        self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, self.conv_dim * 4, 4, 4)
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = self.t_conv3(out)
        out = F.tanh(out)

        return out


cnv_dim = 32
z_size1 = 100
D = Discriminator(cnv_dim)
G = Generator(z_size1, conv_dim=cnv_dim)

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    G.cuda()
    D.cuda()
    print('Training on GPU')
else:
    print('Training on CPU')


def real_loss(D_out, smooth=False):
    batch_size1 = D_out.size(0)
    if smooth:
        labels1 = torch.ones(batch_size1) * 0.9
    else:
        labels1 = torch.ones(batch_size1)
    if train_on_gpu:
        labels1.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels1)
    return loss


lr = 0.002
beta1 = 0.5
beta2 = 0.999
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])


# Training





