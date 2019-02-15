#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --account=def-sdraper
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80

# NOTE:
# array=1-3 will execute the whole script three times.
# nodes=5 means for each array job instance, three nodes are allocated.
# looks like array jobs cannot be executed in same node.
# workaround is to run a for loop as below.

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------

cd /home/s/sdraper/tharindu/projects/decoders/src
pwd
source niagara/setup_env.sh
echo 'executing srun'

#python main.py bec 1200_3_6_rand_ldpc_$SLURM_ARRAY_TASK_ID SPA --codeword=0 --data-dir=$SCRATCH --params .5 .475 .45 .425 .4 .375 .35 .325 .3 .275 .25 .225 .2 .175 .15 .125 .1

for i in `seq 1 10`; do python main.py bec 1200_3_6_rand_ldpc_$i SPA --codeword=0 --data-dir=$SCRATCH --params .5 .475 .45 .425 .4 .375 .35 .325 .3 .275 .25 .225 .2 .175 .15 .125 .1 & done

wait
echo 'done srun!'

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------