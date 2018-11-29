import torch
from torch import nn, optim
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, Nd, Np, in_channel):
        super(Discriminator, self).__init__()
        network_D = [
            nn.Conv2d(in_channel, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)), 
            nn.Conv2d(128, 128, 3, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1),
        ]
        self.layers = nn.Sequential(*network_D)
        self.fc = nn.Linear(320, Nd+Np+1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(0, 0.02)
            elif isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.02)

    def forward(self, input):
        x = self.layers(input)
        x = x.view(-1, 320)
        x = self.fc(x)
        return  x

class Crop(nn.Module):
    def __init__(self):
        super(Crop, self).__init__()

    def forward(self, x):
        B, C, H, W = x.size()
        x = x[:,:, :H-1, :W-1]
        return x

class Generator(nn.Module):
    def __init__(self, in_channel, Np, Nz):
        super(Generator, self).__init__()
        network_G_encoder = [
            nn.Conv2d(in_channel, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(128, 128, 3, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1),
        ]
        self.G_enc = nn.Sequential(*network_G_encoder)

        network_G_decoder = [
            nn.ConvTranspose2d(320, 160, 3, 1, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop(),
            nn.ConvTranspose2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192, 3, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            Crop(),
            nn.ELU(),
            nn.ConvTranspose2d(192, 96, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop(),
            nn.ConvTranspose2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop(),
            nn.ConvTranspose2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, in_channel, 3, 1, 1, bias=False),
            nn.Tanh(),
        ]
        self.G_dec = nn.Sequential(*network_G_decoder)
        self.bridge = nn.Linear(320+Np+Nz, 320*6*6)
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data.normal_(0, 0.02)
            elif isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.02)
            elif isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(0, 0.02)

    def forward(self, input, pose_one_hot, noise):
        x = self.G_enc(input)
        x = x.view(-1, 320)
        x = torch.cat([x, pose_one_hot, noise], 1)
        x = self.bridge(x)
        x = x.view(-1, 320, 6, 6)
        x = self.G_dec(x)
        return x
        

if __name__ == "__main__":
    D = Discriminator(11, 180, 1)
    G = Generator(1, 180, 50)
    print(D)
    print
    print(G)