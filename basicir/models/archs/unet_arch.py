import torch
import torch.nn as nn
from basicir.utils import get_root_logger


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Standard U-Net architecture"""
    def __init__(self, inp_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling
        in_channels_down = inp_channels
        for feature in features:
            self.downs.append(DoubleConv(in_channels_down, feature))
            in_channels_down = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling and save skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse list

        # Upsampling and concatenate with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Handle cases where input dimensions aren't perfectly divisible by 2
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def init_weights(self, pretrained=None, strict=True):
        """Initialize network weights.
        Args:
            pretrained (str | None): Path to pre-trained weights.
            strict (bool): Whether to strictly enforce that the keys
                in pretrained_dict match keys in model.
        """
        logger = get_root_logger()
        if isinstance(pretrained, str):
            try:
                # Load the checkpoint
                checkpoint = torch.load(pretrained, map_location='cpu')
                
                # If the checkpoint contains 'state_dict', use that
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Load the state dict into the model
                self.load_state_dict(state_dict, strict=strict)
                logger.info(f'Successfully loaded pretrained model from {pretrained}')
            except Exception as e:
                logger.error(f'Error loading pretrained model from {pretrained}: {e}')
                raise e
        elif pretrained is None:
            # Use default initialization (PyTorch's default init)
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                          f'But received {type(pretrained)}.') 