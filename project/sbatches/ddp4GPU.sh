#!/bin/bash
#SBATCH --job-name=ddp4GPU
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=ddp4GPU.txt
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=END
#SBATCH --mail-user=yl8798@nyu.edu



module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load python/intel/3.8.6

python train_DDP.py --upscale_factor 4 --cuda --epochs 100 --bs 128 --savetag 4GPU128BS100epochDDP
python train_DDP.py --upscale_factor 4 --cuda --epochs 100 --bs 512 --savetag 4GPU512BS100epochDDP  --lr=0.0004