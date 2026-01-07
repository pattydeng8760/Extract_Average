#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=0-01:00
#SBATCH --job-name=TKE_Extract
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-plavoie

source /project/p/plavoie/denggua1/pd_env.sh
python Average_Extract_Run.py
python Average_Extract_Surface.py
python Extract_PIV.py




