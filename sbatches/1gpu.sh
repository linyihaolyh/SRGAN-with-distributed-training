#!/bin/bash
#SBATCH --job-name=1GPU
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --time=04:30:00
#SBATCH --output=1GPU.txt
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END
#SBATCH --mail-user=yl8798@nyu.edu

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load python/intel/3.8.6


python train.py --upscale_factor 4 --cuda --epochs 100 --bs 128  --savetag 1GPU128biglrBS100epoch --lr=0.0008
python train.py --upscale_factor 4 --cuda --epochs 100 --bs 16  --savetag 1GPU16BS100epoch  --lr=0.0001

