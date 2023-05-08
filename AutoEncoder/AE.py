import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
import os

data = MNIST('./data', transform=ToTensor())
batch_size = 100
img_size = 28 * 28
dl = DataLoader(data, batch_size=batch_size, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(img_size, 14 * 14),
                                     nn.ReLU(),
                                     nn.Linear(14 * 14, 7 * 7),
                                     nn.ReLU(),
                                     nn.Linear(7 * 7, 5 * 5),
                                     nn.ReLU(),
                                     nn.Linear(5 * 5, 10),
                                     nn.ReLU(),
                                     nn.Linear(10, 4)
                                     )

        self.decoder = nn.Sequential(nn.Linear(4, 10),
                                     nn.ReLU(),
                                     nn.Linear(10, 5 * 5),
                                     nn.ReLU(),
                                     nn.Linear(5 * 5, 7 * 7),
                                     nn.ReLU(),
                                     nn.Linear(7 * 7, 14 * 14),
                                     nn.ReLU(),
                                     nn.Linear(14 * 14, img_size),
                                     nn.Sigmoid()
                                     )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0019)

seeds = None
for img, _ in dl:
    seeds = img.reshape(-1, img_size)
    break


def save_images(ii):
    images = model(seeds) if ii > 0 else seeds
    images = images.reshape(images.size(0), 1, 28, 28)
    name = 'test_img_{0:0=4d}.png'.format(ii)
    save_image(images, os.path.join('./samples', name), nrow=10)
    print('save ' + name)


save_images(0)
save_images(1)
total_step = len(dl)

for i in range(300):
    k = 0
    for img, _ in dl:
        img = img.reshape(-1, img_size)
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if k % 200 == 0:
            print(f"step: {k} / {total_step} | epoch: {i} | loss: {loss.item()}")
        k += 1
    save_images(i + 2)
