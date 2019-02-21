#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-sdraper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80

# NOTE:
# S B ATCH --array=1-3 will execute the whole script three times.
# nodes=5 means for each array job instance, three nodes are allocated.
# looks like array jobs cannot be executed in same node.
# workaround is to run a for loop as below.

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
if ! [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."; fi
echo ""
# ---------------------------------------------------------------------

cd /home/s/sdraper/tharindu/projects/decoders/src
pwd
source niagara/setup_env.sh

CASES=("BEC" "BSC_MSA" "AWGN_MSA" "BSC_SPA" "AWGN_SPA")
for CASE in ${CASES[@]}; do
    echo 'Executing srun, CASE:'$CASE
    ./run_sims.sh $CASE $SCRATCH &
done

wait
echo ""
echo "Job finished with exit code $? at: `date`."
echo ""
