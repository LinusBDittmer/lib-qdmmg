#!/bin/bash
#SBATCH -o protocol/send_%j.out
#SBATCH -e protocol/send_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=20gb
#SBATCH --ntasks=2

source /export/home/ccprak16/.bashrc
source /export/home/ccprak16/anaconda3/etc/profile.d/conda.sh
conda activate lib-qdmmg

export OMP_NUM_THREADS=2
python 12-h2of-anion.py >> ./bashout/12-h2of-anion.txt
