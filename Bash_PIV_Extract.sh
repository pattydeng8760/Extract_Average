#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=0-06:59
#SBATCH --job-name=U50_A10_Reynolds_Decomp
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-plavoie

source /project/p/plavoie/denggua1/pd_env.sh
# python Average_Extract_Run.py
# python Average_Extract_Surface.py
# python Extract_PIV.py
python Average_Extract_Reynolds_Decomp.py




