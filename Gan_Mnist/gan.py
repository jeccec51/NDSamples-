import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

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

lr = 0.002

D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)


def real_loss(D_out, smooth=False):
    batch_size_output = D_out.size(0)
    if smooth:
        label = torch.ones(batch_size_output) * 0.9
    else:
        label = torch.ones(batch_size_output)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), label)
    return loss


def fake_loss(D_out):
    batch_size_op = D_out.size(0)
    label = torch.zeros(batch_size_op)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), label)
    return loss


d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)

num_eopchs = 100
samples = []
losses = []

print_every = 400

# Get Some Fixed Data for sampling

sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# Train Network
D.train()
G.train()

for epoch in range(num_eopchs):
    for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images * 2 - 1
        # Train Discriminator
        d_optimizer.zero_grad()
        D_real = D(real_images)
        d_real_loss = real_loss(D_real, smooth=True)
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        # Train with fake image and lipped labels
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)

        # Compute Discriminator losses on fake images using flipped labels
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake)
        g_loss.backward()
        g_optimizer.step()

        # Print loss status
        if batch_i % print_every == 0:
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_eopchs, d_loss.item(), g_loss.item()))
    losses.append((d_loss.item(), g_loss.item()))
    G.eval()
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()

with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

# Plot loss
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discrimnator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()


# Function for viewing list of passed samples

def view_samples(epoch_i, image_samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), image_samples[epoch_i]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')


with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
view_samples(-1, samples)

rows = 10  # split epochs into 10, so 100/10 = every 10 epochs
cols = 6
fig1, axes1 = plt.subplots(figsize=(7, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples) / rows)], axes1):
    for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
        img = img.detach()
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

# randomly generated, new latent vectors
sample_size = 16
rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
rand_z = torch.from_numpy(rand_z).float()

G.eval()  # eval mode
# generated samples
rand_images = G(rand_z)

# 0 indicates the first set of samples in the passed in list
# and we only have one batch of samples, here
view_samples(0, [rand_images])
