#!/bin/bash

#Usage - local  : ./run_sims.sh SEQL HMG --data-dir=./data --console
#Usage - niagara: ./run_sims.sh PARA HMG --data-dir=$SCRATCH --console


PARALLEL=${1:-""}
CASE=$2
OTHER=${@:3:99}  # for extra arguments

EXEC="python -u src/main.py"
CGEN="python -u simulations.py"

log () { echo "run|$CASE|$1"; }
run () { if ! [ "$PARALLEL" == "PARA" ]; then log ">> $1"; eval $1; else log ">> $1 &"; eval $1 & fi }
exc () { run "$EXEC $1"; }

ARR=()
while IFS= read -r line; do
    ARR+=( "$line" )
done < <( eval "$CGEN $CASE $OTHER" )
for line in "${ARR[@]}"; do exc "${line}"; done

log "Waiting..."
wait
log "Done!"
