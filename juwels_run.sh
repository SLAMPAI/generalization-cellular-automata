#!/bin/bash -x
#SBATCH --account=ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=PARTITION
#SBATCH --output=out
#SBATCH --error=err

module purge
module use Stages/Devel-2020
module load GCC/9.3.0
module load TensorFlow/2.3.1-Python-3.8.5
srun python run.py --gen_level simple
