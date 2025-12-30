
import torch
from models.aggregators.salad import SALAD
from models.aggregators.salad_base import SALADBase
from models.aggregators.cross_image_encoder import CrossImageEncoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Instantiate models
    # Using default parameters from __init__ (now optimized)
    salad_full = SALAD(token_dim=256) # Ensure token_dim matches
    salad_base = SALADBase(token_dim=256)
    
    # Cross encoder is now initialized with input_dim=256 inside SALAD
    # We can check standalone too
    cross_encoder = CrossImageEncoder(input_dim=256)

    print(f"Counting parameters for Optimized Salad architectures...")
    
    # 1. Salad Base (No Cross Image)
    params_base = count_parameters(salad_base)
    print(f"Salad Base (No Cross Image): {params_base:,} parameters")

    # 2. Salad Full (With Cross Image)
    params_full = count_parameters(salad_full)
    print(f"Salad Full (With Cross Image): {params_full:,} parameters")
    
    # 3. Cross Image Encoder only
    params_cross = count_parameters(cross_encoder)
    print(f"Cross Image Encoder only: {params_cross:,} parameters")

    # Verify difference
    diff = params_full - params_base
    print(f"Difference (Full - Base): {diff:,} parameters")
    
    if diff == params_cross:
        print("Verification Successful: Difference equals Cross Image Encoder parameters.")
    else:
        print("Verification Failed: Difference does NOT match Cross Image Encoder parameters.")
        print(f"Discrepancy: {diff - params_cross:,}")

if __name__ == "__main__":
    main()
