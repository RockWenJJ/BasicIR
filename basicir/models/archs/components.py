import torch
import torch.nn as nn
from torch import Tensor
import kornia
from typing import Dict, List, Tuple, Union, Optional

class CompressedHE(nn.Module):
    def __init__(self, compression_factor: float = 0.5):
        super(CompressedHE, self).__init__()
        self.compression_factor = compression_factor
        
    def forward(self, x: Tensor) -> Tensor:
        # Ensure the input is in the correct format
        assert x.dim() == 4, "Input tensor must have 4 dimensions (batch, channel, height, width)"
        
        # Scale input to range (0, 255) for histogram calculation
        x = (x * 255).to(torch.int32)
        
        # Initialize the output tensor
        output = torch.empty_like(x, dtype=torch.float32)
        
        # Process each image separately
        for i in range(x.size(0)):
            im = x[i, :, :, :]
            im = self.equalize_single(im)
            output[i, :, :, :] = im
        
        # Scale output back to range (0, 1)
        output = output.to(torch.float32) / 255.0
        
        return output
    
    def scale_channel(self, im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[c, :, :]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)#.type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1).to(lut.device), lut[:-1]]) 
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im)
        return result
    
    def equalize_single(self, im):
        s1 = self.scale_channel(im, 0)
        s2 = self.scale_channel(im, 1)
        s3 = self.scale_channel(im, 2)
        image = torch.stack([s1, s2, s3], 0)
        return image
    
class EdgeEnhance(nn.Module):
    def __init__(self, low_threshold: float = 0.1, high_threshold: float = 0.2):
        super(EdgeEnhance, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, x: Tensor) -> Tensor:
        # Convert RGB to LAB
        lab = kornia.color.rgb_to_lab(x)
        
        # Extract L channel
        l_channel = lab[:, 0:1, :, :]
        
        # Normalize L channel to range [0, 1]
        l_normalized = (l_channel - l_channel.min()) / (l_channel.max() - l_channel.min())
        
        # Apply Canny edge detection
        magnitudes, edges = kornia.filters.canny(
            l_normalized,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold
        )
        
        # Get min and max values in the last two dimensions (height and width)
        magnitudes_min = magnitudes.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        magnitudes_max = magnitudes.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        
        # Normalize magnitudes
        magnitudes_normalized = (magnitudes - magnitudes_min) / (magnitudes_max - magnitudes_min + 1e-8)

        # Combine the original image with the edge map
        enhanced = x * magnitudes_normalized
        
        return enhanced
