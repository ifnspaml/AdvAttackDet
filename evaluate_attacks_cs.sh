#!/bin/bash

#SBATCH --job-name="Gsusdn Eval Depth"
#SBATCH --time=120:00:00
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=2
#SBATCH --mem=14000
#SBATCH --partition=gpu
#SBATCH --comment="Evaluate the Gsusdn Network"

model_name="train_ssim_all_0003"
eval_name="eval_attack_cs"
p="gaussian"
i=0.0

srun python3 eval_joint.py \
        --sys-best-effort-determinism \
        --model-name ${eval_name} \
        --model-load ${model_name}/checkpoints/epoch_20/ \
        --joint-validation-loaders "cityscapes_seq_validation" \
        --joint-validation-resize-width 640 \
        --joint-validation-resize-height 320 \
        --joint-validation-perturbation-type $p \
        --joint-validation-perturbation-strength $i \
        --eval-run-detection

for p in "gaussian" "saltandpepper" "fgsm" "bim" "pgd" "o-pgd"
do
echo $p

for i in  1.0 2.0 4.0 8.0 16.0 32.0
do
echo $i

srun python3 eval_joint.py \
        --sys-best-effort-determinism \
        --model-name ${eval_name} \
        --model-load ${model_name}/checkpoints/epoch_20/ \
        --joint-validation-loaders "cityscapes_seq_validation" \
        --joint-validation-resize-width 640 \
        --joint-validation-resize-height 320 \
        --joint-validation-perturbation-type $p \
        --joint-validation-perturbation-strength $i \
        --eval-run-detection \
        --eval-dont-save-threshs

done

done

