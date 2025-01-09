import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    

class LUUIE(nn.Module):
    def __init__(self):
        super(LUUIE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x, training=True):
        if training:
            x = self.encoder(x)
            t = self.decoder_t(x)
            b = self.decoder_b(x)
            j = self.decoder_j(x)
            return j, t, b
        else:
            x = self.encoder(x)
            j = self.decoder
            return j