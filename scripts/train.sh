#!/bin/bash

# =============================================================================
# Training script for SALAD with Cross-Image Learning
# =============================================================================

# Option draft: Train with frozen base
python train_cross_image.py \
    --freeze_base \
    --batch_size 20 \
    --img_per_place 4 \
    --epochs 15 \
    --lr 6e-5 \
    --num_trainable_blocks 4 \
    --image_size 224 \
    --val_sets msls_val \
    --log_dir ./logs/cross_image_frozen/

# Option 1: Train with frozen base + pretrained weights
# python train_cross_image.py \
#     --pretrained_path pretrainedWeight/Salad/last.ckpt \
#     --freeze_base \
#     --batch_size 20 \
#     --img_per_place 4 \
#     --epochs 15 \
#     --lr 6e-5 \
#     --num_trainable_blocks 4 \
#     --image_size 224 \
#     --val_sets msls_val \
#     --log_dir ./logs/cross_image_frozen/

# Option 2: Fine-tune entire model
# python train_cross_image.py \
#     --pretrained_path pretrainedWeight/Salad/last.ckpt \
#     --batch_size 20 \
#     --img_per_place 4 \
#     --epochs 4 \
#     --lr 1e-5 \
#     --image_size 224 \
#     --val_sets msls_val \
#     --log_dir ./logs/cross_image_finetune/

# Option 3: Train from scratch
# python train_cross_image.py \
#     --pretrained_path "" \
#     --batch_size 20 \
#     --img_per_place 4 \
#     --epochs 10 \
#     --lr 6e-5 \
#     --image_size 224 \
#     --val_sets msls_val \
#     --log_dir ./logs/cross_image_scratch/

# =============================================================================
# Evaluation
# =============================================================================

# python evaluate.py \
#     --checkpoint ./logs/cross_image_frozen/cross_image_salad/checkpoints/last.ckpt \
#     --val_sets msls_val \
#     --batch_size 32
