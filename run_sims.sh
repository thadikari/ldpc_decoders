#!/usr/bin/env bash

#run sequencially: ./run_sims.sh BEC ../temp
#run in parallel:  ./run_sims.sh BEC ../temp PARA


CASE=$1
DATA_DIR=$2
PARALLEL=$3

log () { echo "run_sims[$CASE]: $1"; }

run_sim_1 () {
    local CHANNEL=$1
    local ARGS="$2 --data-dir=$DATA_DIR --min-wec=$MIN_WEC"
    local DEFAULT_MAX_ITER=$3
    declare -a MAX_ITER_ARR=("${!4}")
    local DEFAULT_ARGS="$ARGS --max-iter=$DEFAULT_MAX_ITER"    

    CMDS=()
    for i in `seq 1 10`; do CMDS+=("python -u main.py $CHANNEL 1200_3_6_rand_ldpc_$i $DEFAULT_ARGS"); done
    CMDS+=("python -u main.py $CHANNEL 1200_3_6_ldpc $DEFAULT_ARGS")
    for i in ${MAX_ITER_ARR[@]}; do CMDS+=("python -u main.py $CHANNEL 1200_3_6_ldpc $ARGS --max-iter=$i"); done
    
    for i in `seq 1 ${#CMDS[@]}`; do
        cm=${CMDS[$i-1]}
        if [ -z "$PARALLEL" ]; then log "$cm"; eval $cm; else log "$cm &"; eval $cm & fi
    done
}

MIN_WEC=100
ARR=(1 2 3 6 40 100)

case $CASE in
    "BEC")
        run_sim_1 bec "SPA --codeword=0 --params .475 .45 .425 .4 .375 .35 .325 .3 .275 .25" 10 ARR[@]
        ;;
    "BSC_MSA")
        run_sim_1 bsc "MSA --codeword=1 --params .081 .071 .061 .051 .0451 .031 .0251 .021 .0151 .011" 10 ARR[@]
        ;;
    "AWGN_MSA")
        run_sim_1 biawgn "MSA --codeword=0 --params .5 .75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3.0" 10 ARR[@]
        ;;
    "BSC_SPA")
        run_sim_1 bsc "SPA --codeword=0 --params .1 .09 .08 .07 .06 .05 .04" 10 ARR[@]
        ;;
    "AWGN_SPA")
        run_sim_1 biawgn "SPA --codeword=0 --params .5 .75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3.0" 10 ARR[@]
        ;;
    *)
        log "Non-existent CASE=$CASE!"
        exit -1
        ;;
esac

wait
log "run_sims: CASE=$CASE Done!"
