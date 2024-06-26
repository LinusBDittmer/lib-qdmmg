#!/bin/bash
#SBATCH -o gridcheck_%j.out
#SBATCH -e gridcheck_%j.err
#SBATCH --time=120:00:00
#SBATCH --mem=150gb
#SBATCH --ntasks=4

#Job details
export SLURMVARS=$(env | grep 'SLURM')
echo "---  JOB DETAILS  ---"
echo "Slurm variables: "
for svar in $SLURMVARS; do echo $svar; done
echo "---"
echo 
echo

#Setting up directories
echo "Setting up Directories..."
export HOME_DIR=$SLURM_SUBMIT_DIR
export SCRATCH_DIR="/scratch/ldittmer/gridcheck_id"$SLURM_JOB_ID
echo "Creating relevant scratch directories"
mkdir -p $SCRATCH_DIR

#Defining Python variables
echo "Copying Python script to scratch"
cp $HOME_DIR/gridcheck.py $SCRATCH_DIR
export PYEXEC=$SCRATCH_DIR/gridcheck.py
export PYSCF_TMPDIR=$SCRATCH_DIR
#export PYSCF_MAX_MEMORY=150000

echo "Preparing Python execution"
source /export/home/ldittmer/.bashrc
source /export/home/ldittmer/miniconda3/etc/profile.d/conda.sh
conda activate lib-qdmmg
export OMP_NUM_THREADS=4,1
export MKL_NUM_THREADS=4

echo "Running Python script $PYEXEC"
cd $SCRATCH_DIR
python $PYEXEC > $PYEXEC.out
cd $HOME_DIR
echo "Finished Running Python script"

cp $PYEXEC.out $HOME_DIR
#rm -r $SCRATCH_DIR
echo "Finished"
