import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim


num_workers = 0
batch_size = 64
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
img = np.squeeze(images[0])

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')


class Discriminator(nn.Module):

    def __init__(self, inp_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.3)
        x = self.dropout(x)
        out = self.fc4(x)
        return out


class Generator(nn.Module):

    def __init__(self, input_dim, hidden_size, output_size):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.fc4 = nn.Linear(hidden_size * 4, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = F.tanh(self.fc4(x))
        return out


# Discriminator Hyper Parameters
input_size = 784
d_output_size = 1
d_hidden_size = 32

# Generator Hyper Parameters
z_size = 100
g_output_size = 784
g_hidden_size = 32

D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)


def real_loss(D_out, smooth=False):
    batch_size_output = D_out.size(0)
    if smooth:
        label = torch.ones(batch_size_output) * 0.9
    else:
        label = torch.ones(batch_size_output)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeez(), label)
    return loss


def fake_loss(D_out):
    batch_size_op = D_out.size(0)
    label = torch.zeros(batch_size_op)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), label)
    return loss




print(D)
print()
print(G)

