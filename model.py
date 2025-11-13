import torch
from torch import nn

from utils import crop


class u_net(nn.Module):
    def __init__(self, in_size=572, in_channel=1):
        super().__init__()
        self.in_size = in_size
        self.in_channel = in_channel
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU()
        )
        self.down5up0 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.ConvTranspose2d(1024, 512, 2, 2)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2)
        )
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, X):
        copy1 = self.down1(X)
        copy2 = self.down2(copy1)
        copy3 = self.down3(copy2)
        copy4 = self.down4(copy3)

        low = self.down5up0(copy4)
        crop4 = crop(copy4, low.shape)
        cat1 = torch.cat((crop4, low), dim=1)
        up1 = self.up1(cat1)

        crop3 = crop(copy3, up1.shape)
        cat2 = torch.cat((crop3, up1), dim=1)
        up2 = self.up2(cat2)

        crop2 = crop(copy2, up2.shape)
        cat3 = torch.cat((crop2, up2), dim=1)
        up3 = self.up3(cat3)

        crop1 = crop(copy1, up3.shape)
        cat4 = torch.cat((crop1, up3), dim=1)

        output = self.final(cat4)
        return output


if __name__ == "__main__":
    net = u_net()
    X = torch.empty(size=(1, 1, 572, 572))
    print(net(X).shape)
