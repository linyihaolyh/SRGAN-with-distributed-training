#!/bin/bash
#SBATCH --job-name=lightning4GPU
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=lightning4GPU.txt
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mail-type=END
#SBATCH --mail-user=yl8798@nyu.edu
#SBATCH --cpus-per-task=20

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load python/intel/3.8.6


python ddp_lightning.py --bs 128 --epochs 100 --device 4
python ddp_lightning.py --bs 512 --epochs 100 --device 4