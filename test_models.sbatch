#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=END
#SBATCH --gres=gpu:2
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.6.0.64
srun python test_models.py 2

