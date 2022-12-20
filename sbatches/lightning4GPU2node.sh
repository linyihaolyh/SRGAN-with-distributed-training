#!/bin/bash
#SBATCH --job-name=lightning4GPU2node
#SBATCH --nodes=2
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=lightning4GPU2node.txt
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --ntasks-per-node=2
#SBATCH --mail-type=END
#SBATCH --mail-user=yl8798@nyu.edu

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load python/intel/3.8.6


srun python ddp_lightning.py --bs 128 --epochs 20 --device 2 --nodes 2
