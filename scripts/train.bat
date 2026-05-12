@echo off
REM =============================================================================
REM Training script for SALAD with Cross-Image Learning (Windows)
REM =============================================================================

echo.
echo ============================================================
echo SALAD Cross-Image Training
echo ============================================================
echo.

REM Option 1: Train with frozen base (only train CrossImageEncoder)
python train_cross_image.py ^
    --pretrained_path pretrainedWeight/Salad/last.ckpt ^
    --freeze_base ^
    --batch_size 20 ^
    --img_per_place 4 ^
    --epochs 4 ^
    --lr 6e-5 ^
    --log_dir ./logs/cross_image_frozen/

REM Option 2: Fine-tune entire model (remove REM to use)
REM python train_cross_image.py ^
REM     --pretrained_path pretrainedWeight/Salad/last.ckpt ^
REM     --batch_size 20 ^
REM     --img_per_place 4 ^
REM     --epochs 4 ^
REM     --lr 1e-5 ^
REM     --log_dir ./logs/cross_image_finetune/

pause
