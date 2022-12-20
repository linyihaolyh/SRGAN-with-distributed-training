#!/bin/bash
#SBATCH --job-name=ddp2GPU
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --time=03:30:00
#SBATCH --output=ddp2GPU.txt
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --ntasks-per-node=2
#SBATCH --mail-type=END
#SBATCH --mail-user=yl8798@nyu.edu



module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load python/intel/3.8.6

python train_DDP.py --upscale_factor 4 --cuda --epochs 100 --bs 128 --savetag 2GPU128BS100epochDDP
python train_DDP.py --upscale_factor 4 --cuda --epochs 100 --bs 512 --savetag 2GPU512BS100epochDDP  --lr=0.0004