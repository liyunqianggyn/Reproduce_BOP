#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=8:20:00
#SBATCH --ntasks=1
#SBATCH --mem=2696
#SBATCH --gres=gpu

export PATH=~/anaconda3/bin:$PATH

srun python < trainer.py
