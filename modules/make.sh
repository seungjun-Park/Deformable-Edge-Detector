#!/bin/bash

#SRUN --job-name Build
#SRUN --gres=gpu:1
#SRUN --cpus-per-gpu=8
#SRUN --mem-per-gpu=29G
#SRUN --time 1-0
#SRUN --partition debug_ugrad
#SRUN -o slurm/logs/%A-%x.out
# srun --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=29G -p debug_ugrad -w ariel-v10 --pty $SHELL

python setup.py build install

exit 0