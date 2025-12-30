import torch
import torch.nn as nn

from .salad_base import SALADBase
from .cross_image_encoder import CrossImageEncoder


class SALAD(nn.Module):
    """
    Full SALAD with optional cross-image learning.
    
    Combines SALADBase (pretrained) with CrossImageEncoder (trainable).
    
    Usage:
        # Create model
        model = SALAD(num_channels=768, ...)
        
        # Load pretrained base weights
        model.load_base_weights('pretrained_salad.pth')
        model.freeze_base()  # Optional: only train cross_encoder
        
        # Training
        descriptor = model(x)  # Auto uses cross-image when training
        
        # Inference
        query_desc = model.forward_single(x)      # For queries
        db_desc = model.forward_database(x)       # For database with cross-image
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
        img_per_place=4,
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.img_per_place = img_per_place

        # Base SALAD (can load pretrained weights)
        self.base = SALADBase(
            num_channels=num_channels,
            num_clusters=num_clusters,
            cluster_dim=cluster_dim,
            token_dim=token_dim,
            dropout=dropout,
        )

        # Cross-image encoder (now working on global token)
        self.cross_encoder = CrossImageEncoder(
            input_dim=token_dim,
            img_per_place=img_per_place,
            dropout=dropout,
        )

    def load_base_weights(self, checkpoint_path, strict=True):
        """
        Load pretrained weights for SALADBase only.
        CrossImageEncoder remains randomly initialized.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter and remap keys for base module
        base_state_dict = {}
        base_keys = ["token_features", "cluster_features", "score", "dust_bin"]

        for k, v in state_dict.items():
            new_key = k
            # Remove common prefixes
            for prefix in ["aggregator.", "model.aggregator.", "base."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix) :]
                    break

            if any(new_key.startswith(name) for name in base_keys):
                base_state_dict[new_key] = v

        self.base.load_state_dict(base_state_dict, strict=strict)
        print(f"Loaded {len(base_state_dict)} base weights from {checkpoint_path}")

    def load_cross_encoder_weights(self, checkpoint_path, strict=True):
        """Load pretrained weights for CrossImageEncoder only."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        cross_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["cross_encoder.", "model.cross_encoder."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix) :]
                    break

            if new_key.startswith("encoder"):
                cross_state_dict[new_key] = v

        self.cross_encoder.load_state_dict(cross_state_dict, strict=strict)
        print(f"Loaded {len(cross_state_dict)} cross-encoder weights")

    def freeze_base(self):
        """Freeze base SALAD weights, only train cross-image encoder."""
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze base SALAD weights."""
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x, use_cross_image=True):
        """
        Forward pass with optional cross-image learning.
        
        Cross-image is applied only when:
        - use_cross_image=True
        - model.training=True  
        - batch_size is divisible by img_per_place
        """
        s, t = self.base.compute_features(x)

        if use_cross_image and self.training and s.shape[0] % self.img_per_place == 0:
            # Apply to global token t
            t = self.cross_encoder(t)

        return self.base.build_descriptor(s, t)

    def forward_single(self, x):
        """Forward for single image (query inference) - no cross-image."""
        return self.base(x)

    def forward_single_with_cross_image(self, x):
        """
        Forward for single image with cross-image enhancement.
        Duplicates each image img_per_place times to apply cross-image.
        
        Args:
            x: (features [B, C, H, W], token [B, C])
        Returns:
            descriptor [B, descriptor_dim]
            
        Note: When duplicating the same image, cross-image attention will
        see identical inputs. The output will be consistent with training
        distribution since it goes through the same layers.
        """
        features, token = x
        B = features.shape[0]
        
        # Duplicate each image img_per_place times
        # Use repeat_interleave to keep same image together:
        # [img1, img1, img1, img1, img2, img2, img2, img2, ...]
        features_dup = features.repeat_interleave(self.img_per_place, dim=0)  # [B*img_per_place, C, H, W]
        token_dup = token.repeat_interleave(self.img_per_place, dim=0)  # [B*img_per_place, C]
        
        # Compute base features
        s, t = self.base.compute_features((features_dup, token_dup))
        
        # Apply cross-image
        # Each group of img_per_place will be the same image duplicated
        # Apply to global token t instead of s
        t = self.cross_encoder(t)
        
        # Build descriptor for all duplicates
        descriptors = self.base.build_descriptor(s, t)  # [B*img_per_place, descriptor_dim]
        
        # Reshape and take first (all duplicates should be identical after cross-image)
        descriptors = descriptors.view(B, self.img_per_place, -1)[:, 0, :]  # [B, descriptor_dim]
        
        # Re-normalize after selection
        descriptors = nn.functional.normalize(descriptors, p=2, dim=-1)
        
        return descriptors

    def forward_database(self, x):
        """
        Forward for database images with cross-image enhancement.
        Batch size must be divisible by img_per_place.
        """
        s, t = self.base.compute_features(x)

        if s.shape[0] % self.img_per_place == 0:
            t = self.cross_encoder(t)

        return self.base.build_descriptor(s, t)
