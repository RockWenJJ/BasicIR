import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalModelLoss(nn.Module):
    """Physical model loss for underwater image enhancement.
    
    This loss enforces the physical model of underwater image formation:
    I = J * T + B * (1 - T)
    
    Where:
    - I: Underwater image (input)
    - J: Clear image
    - T: Transmission map
    - B: Backscatter
    - W: White point for color correction
    
    Args:
        backscatter_weight (float): Weight for backscatter consistency loss
        transmission_weight (float): Weight for transmission consistency loss
        white_balance_weight (float): Weight for white balance consistency loss
    """
    
    def __init__(self, backscatter_weight=0.5, transmission_weight=0.5, white_balance_weight=0.2):
        super(PhysicalModelLoss, self).__init__()
        self.backscatter_weight = backscatter_weight
        self.transmission_weight = transmission_weight
        self.white_balance_weight = white_balance_weight
        
    def forward(self, input_img, clear_img, backscatter, transmission, white_point):
        """
        Args:
            input_img (Tensor): Input underwater image (B, 3, H, W)
            clear_img (Tensor): Restored clear image (B, 3, H, W)
            backscatter (Tensor): Predicted backscatter (B, 3, H, W)
            transmission (Tensor): Predicted transmission map (B, 3, H, W)
            white_point (Tensor): Predicted white point (B, 3)
        
        Returns:
            Tensor: Total loss value
        """
        batch_size = input_img.size(0)
        
        # 1. Reconstruction loss (make sure I = J*T + B*(1-T))
        reconstruction = clear_img * transmission + backscatter * (1 - transmission)
        recon_loss = F.l1_loss(reconstruction, input_img)
        
        # 2. Backscatter constraints:
        # - Backscatter should be relatively uniform (small gradient)
        # - Backscatter should be smaller than input image
        back_grad_x = torch.abs(backscatter[:, :, :, :-1] - backscatter[:, :, :, 1:])
        back_grad_y = torch.abs(backscatter[:, :, :-1, :] - backscatter[:, :, 1:, :])
        back_smoothness_loss = (torch.mean(back_grad_x) + torch.mean(back_grad_y)) / 2.0
        
        back_input_diff = F.relu(backscatter - input_img)  # Only penalize if B > I
        back_input_loss = torch.mean(back_input_diff)
        
        backscatter_loss = back_smoothness_loss + back_input_loss
        
        # 3. Transmission constraints:
        # - Transmission should be in [0, 1]
        # - Transmission should be smooth but preserve edges
        transmission_value_loss = torch.mean(F.relu(-transmission) + F.relu(transmission - 1.0))
        
        # Edge-aware smoothness using the gradients of input image as guidance
        input_grad_x = torch.mean(torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:]), dim=1, keepdim=True)
        input_grad_y = torch.mean(torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :]), dim=1, keepdim=True)
        
        # Calculate transmission gradients
        trans_grad_x = torch.abs(transmission[:, :, :, :-1] - transmission[:, :, :, 1:])
        trans_grad_y = torch.abs(transmission[:, :, :-1, :] - transmission[:, :, 1:, :])
        
        # Weight gradients with exponential of negative input gradients to preserve edges
        weights_x = torch.exp(-input_grad_x)
        weights_y = torch.exp(-input_grad_y)
        
        # Apply weights to transmission gradients and calculate loss
        trans_smooth_x = trans_grad_x * weights_x[:, :, :, :-1]
        trans_smooth_y = trans_grad_y * weights_y[:, :, :-1, :]
        trans_smoothness_loss = torch.mean(trans_smooth_x) + torch.mean(trans_smooth_y)
        
        transmission_loss = transmission_value_loss + trans_smoothness_loss
        
        # 4. White point constraints:
        # - White point should be within reasonable bounds (e.g., [0.5, 1.5])
        # - White point should be balanced across channels for natural lighting
        
        # Check white point is within reasonable bounds
        white_bound_loss = torch.mean(
            F.relu(0.5 - white_point) + F.relu(white_point - 1.5)
        )
        
        # White balance should have a balanced ratio across channels
        # We want the ratios between channels to be reasonably close to 1
        r_g_ratio = white_point[:, 0] / (white_point[:, 1] + 1e-6)
        b_g_ratio = white_point[:, 2] / (white_point[:, 1] + 1e-6)
        
        # Penalize if ratios are too far from 1 (natural white balance)
        ratio_loss = torch.mean(
            F.relu(torch.abs(r_g_ratio - 1.0) - 0.2) + 
            F.relu(torch.abs(b_g_ratio - 1.0) - 0.2)
        )
        
        white_balance_loss = white_bound_loss + ratio_loss
        
        # 5. Combine all losses with weights
        total_loss = (
            recon_loss + 
            self.backscatter_weight * backscatter_loss + 
            self.transmission_weight * transmission_loss + 
            self.white_balance_weight * white_balance_loss
        )
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'backscatter_loss': backscatter_loss.item(),
            'transmission_loss': transmission_loss.item(),
            'white_balance_loss': white_balance_loss.item()
        } 