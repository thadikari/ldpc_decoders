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

RUN_ID="$SLURM_ARRAY_JOB_ID/$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
echo ""
echo "Job Array ID / Job ID , Array Task ID / Array Task Count: $RUN_ID"
echo ""
# ---------------------------------------------------------------------

log () { echo "submit|$RUN_ID|$1"; }
run () { log ">> $1"; eval "$1"; }
list () {
    declare -a LIST=("${!1}")
    for CASE in ${LIST[@]}; do run "./run_sims.sh $CASE PARA --data-dir=$SCRATCH" & done
}

run "cd /home/s/sdraper/tharindu/projects/decoders"
run "pwd"
run "source niagara/setup_env.sh"

CASES=("BEC" "BSC_MSA" "BIAWGN_MSA" "BSC_SPA" "BIAWGN_SPA")
list CASES[@]  # TODO: split cases by Array Task Count

log "Waiting..."
wait
log "Job finished with exit code $? at: `date`."
