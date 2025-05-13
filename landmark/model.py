import torch
import torch.nn as nn


class Discriminator(nn.Module):
    '''
    Input: VGG features of size
    '''
    def __init__(self, model_config):
        super(Discriminator, self).__init__()
        
        channels = model_config.channels
        mlp = []
        for i in range(1, len(channels)):
            mlp.append(nn.Sequential(
                nn.Linear(channels[i-1], channels[i], bias=False),
                nn.BatchNorm1d(channels[i]),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.mlp = nn.Sequential(*mlp)
        self.fc = nn.Linear(model_config.channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        return self.sigmoid(self.fc(x))
    

class LmkDistModel(nn.Module):

    def __init__(self, model_config) -> None:
        super().__init__()

        channels = model_config.channels
        self.landmark_num = model_config.landmark_num
        self.in_channels = self.landmark_num * 3
        self.out_channels = self.landmark_num * 3

        self.enc_lmk = nn.Sequential(
            nn.Linear(self.in_channels, channels[0] // 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc_delta = nn.Sequential(
            nn.Linear(self.in_channels, channels[0] // 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        mlp = []
        for i in range(1, len(channels)):
            mlp.append(nn.Sequential(
                nn.Linear(channels[i-1], channels[i], bias=False),
                nn.LeakyReLU(inplace=True),
            ))
        self.mlp = nn.Sequential(*mlp)
        self.fc = nn.Linear(channels[-1], self.out_channels)

    def forward(self, lmk, delta):
        
        lmk = self.enc_lmk(lmk)
        delta = self.enc_delta(delta)
        x = torch.cat([lmk, delta], dim=-1)
        x = self.mlp(x)
        x = self.fc(x)
        
        return x
