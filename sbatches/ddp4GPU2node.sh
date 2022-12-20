#!/bin/bash
#SBATCH --job-name=ddp4GPU2node
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=ddp4GPU2node.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=yl8798@nyu.edu



module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load python/intel/3.8.6

srun python train_DDP.py --upscale_factor 4 --cuda --epochs 20 --bs 128 --savetag 4GPU128BS100epochDDP
