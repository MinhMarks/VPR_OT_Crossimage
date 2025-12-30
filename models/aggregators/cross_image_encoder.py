import torch
import torch.nn as nn


class CrossImageEncoder(nn.Module):
    """
    Cross-image Transformer encoder for learning relationships between
    multiple images from the same place.
    
    This module enhances features by allowing images from the same
    place to share information through self-attention.
    """

    def __init__(self, input_dim=256, img_per_place=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.img_per_place = img_per_place

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=input_dim * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Apply cross-image attention enhancement.
        
        Args:
            x: [B, input_dim] where B = num_places * img_per_place
        Returns:
            x_enhanced: [B, input_dim] with residual connection
        """
        B, D = x.shape
        num_places = B // self.img_per_place
        
        # Reshape for Transformer: [Sequence Length, Batch Size, Embedding Dim]
        # In our case: Sequence Length = img_per_place
        # Batch Size = num_places
        
        # [B, D] -> [num_places, img_per_place, D]
        x_place = x.view(num_places, self.img_per_place, D)
        
        # Permute to [img_per_place, num_places, D] for Transformer (batch_first=False)
        x_seq = x_place.permute(1, 0, 2)

        # Cross-image attention
        x_encoded = self.encoder(x_seq)

        # Reshape back to [B, D]
        # [img_per_place, num_places, D] -> [num_places, img_per_place, D]
        x_encoded = x_encoded.permute(1, 0, 2)
        x_encoded = x_encoded.contiguous().view(B, D)
        
        x_encoded = nn.functional.normalize(x_encoded, p=2, dim=1)

        return x + x_encoded  # Residual connection
