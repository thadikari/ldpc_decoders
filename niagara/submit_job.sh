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
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------

cd /home/s/sdraper/tharindu/projects/decoders/src
pwd
source niagara/setup_env.sh
echo 'executing srun'


# sleep 12h  # hold the node


##### BEC
#python main.py bec 1200_3_6_rand_ldpc_$SLURM_ARRAY_TASK_ID SPA --codeword=0 --data-dir=$SCRATCH --params .5 .475 .45 .425 .4 .375 .35 .325 .3 .275 .25 .225 .2 .175 .15 .125 .1

#for i in `seq 1 10`; do python main.py bsc 1200_3_6_rand_ldpc_$i MSA --codeword=0 --data-dir=$SCRATCH --max-iter=100 --params .475 .45 .425 .4 .375 .35 .325 .3 .275 .25 .225 .2 .175 .15 .125 .1 & done



##### BSC MSA
#for i in `seq 1 10`; do python -u main.py bsc 1200_3_6_rand_ldpc_$i MSA --codeword=0 --data-dir=$SCRATCH --min-wec=100 --max-iter=100 --params .08 .07 .06 .05 .045 .04 .035 .03 .025 .02 .015 .01 & done

#python -u main.py bsc 1200_3_6_ldpc MSA --codeword=0 --data-dir=$SCRATCH --min-wec=100 --max-iter=100 --params .08 .07 .06 .05 .045 .04 .035 .03 .025 .02 .015 .01 --console &

#for i in 1 2 3 6 10 40; do python main.py bsc 1200_3_6_ldpc MSA --codeword=0 --data-dir=$SCRATCH --max-iter=$i --params .08 .07 .06 .05 .045 .04 .035 .03 .025 .02 & done



##### AWGN MSA
#for i in `seq 1 10`; do python main.py biawgn 1200_3_6_rand_ldpc_$i MSA --codeword=0 --data-dir=$SCRATCH --max-iter=100 --params .5 .75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3.0 & done



##### BSC SPA
#for i in `seq 1 10`; do python -u main.py bsc 1200_3_6_rand_ldpc_$i SPA --codeword=0 --data-dir=$SCRATCH --min-wec=100 --max-iter=100 --params .1 .09 .08 .07 .06 .05 & done

#python -u main.py bsc 1200_3_6_ldpc SPA --codeword=0 --data-dir=$SCRATCH --min-wec=100 --max-iter=100 --params .04 &



##### AWGN SPA
#for i in `seq 1 10`; do python -u main.py biawgn 1200_3_6_rand_ldpc_$i SPA --codeword=0 --data-dir=$SCRATCH --min-wec=100 --max-iter=100 --params .5 .75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3.0 & done

#python -u main.py biawgn 1200_3_6_ldpc SPA --codeword=0 --data-dir=$SCRATCH --min-wec=100 --max-iter=100 --params .5 .75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3.0 &

wait
echo 'done srun!'

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------