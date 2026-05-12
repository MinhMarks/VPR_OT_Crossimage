@echo off
REM =============================================================================
REM Evaluation script for SALAD with Cross-Image Learning (Windows)
REM =============================================================================

echo.
echo ============================================================
echo SALAD Model Evaluation
echo ============================================================
echo.

REM Set checkpoint path (modify this to your trained model)
set CHECKPOINT=./logs/cross_image_frozen/cross_image_salad/freeze_True_lr_6e-05/checkpoints/last.ckpt

REM Evaluate on MSLS validation set
python evaluate.py ^
    --checkpoint %CHECKPOINT% ^
    --val_sets msls_val ^
    --batch_size 32

REM Evaluate on Pittsburgh (uncomment to use)
REM python evaluate.py ^
REM     --checkpoint %CHECKPOINT% ^
REM     --val_sets pitts30k_val,pitts30k_test ^
REM     --batch_size 32

pause
