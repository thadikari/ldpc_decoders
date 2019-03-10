#!/bin/bash

#Usage: ./plot_results.sh HMG png ./data ./plots --silent --error=ber


CASE=${1}
EXT=${2:-png}
DATA_DIR=${3:-"./data"}
PLOTS_DIR=${4:-"./plots"}
OTHER="$5 $6 $7 $8"  # for --silent OR --error=ber

DIRS="--data-dir=$DATA_DIR --plots-dir=$PLOTS_DIR"
EXEC="python -u src/graph.py"

log () { echo "plot|$CASE|$1"; }
run () { echo ">> $1"; eval "$1"; }
exc () {
    SAVE="--save ${CASE}_${2}.$EXT"
    run "$EXEC $1 $DIRS $SAVE $OTHER";
}

plot_1 () {
    local CHANNEL=$1
    local DECODER=$2
    local ARGS1=$3
    local ARGS2=$4
    local ARGS3=$5

    run "python -u src/stats.py $CHANNEL 1200_3_6_rand_ldpc $DECODER --data-dir=$DATA_DIR"
    exc "$CHANNEL 1200_3_6_rand_ldpc $DECODER ensemble $ARGS1" "ensemble"
    exc "$CHANNEL 1200_3_6_rand_ldpc $DECODER compare $ARGS2 --extra 1200_3_6_ldpc" "compare"
    exc "$CHANNEL 1200_3_6_ldpc $DECODER max_iter $ARGS3" "max_iter"
}

list () {
    declare -a LIST=("${!1}")
    for it in ${LIST[@]}; do run "$0 ${it} ${EXT} ${DATA_DIR} ${PLOTS_DIR} ${OTHER}"; done
}

case ${CASE} in
    "HMG") # all hamming code sims
        exc "bec 7_4_hamming ML SPA LP ADMM comp_dec --error ber" "BEC"
        exc "bsc 7_4_hamming ML SPA MSA LP ADMM comp_dec --error ber" "BSC"
        exc "biawgn 7_4_hamming ML SPA MSA LP ADMM comp_dec --error ber" "BIAWGN"
        ;;
    "MAR") # margulis code sims
        CONFIG="margulis ADMM single"
        exc "biawgn $CONFIG" "BIAWGN"
        exc "bec $CONFIG" "BEC"
        exc "bsc $CONFIG" "BSC"
        ;;
    "BEC")
        plot_1 bec SPA "--xlim .3 .5 --ylim 2e-7 .5 --max-iter=10" "--max-iter=10 --xlim .3 .5 --ylim 3e-5 .5" ""
        ;;
    "BSC_MSA")
        plot_1 bsc MSA "--xlim 0.02 0.08 --ylim 6e-6 .2 --max-iter=10" "--max-iter=10 --xlim 0.015 0.08" ""
        ;;
    "BIAWGN_MSA")
        plot_1 biawgn MSA "--xlim .5 3 --ylim 3e-5 .2 --max-iter=10" "--max-iter=10 --xlim .5 3 --ylim 3e-5 .2" "--xlim .5 3 --ylim 4e-4 .2"
        ;;
    "BSC_SPA")
        plot_1 bsc SPA "--max-iter=10" "--max-iter=10" ""
        ;;
    "BIAWGN_SPA")
        plot_1 biawgn SPA "--xlim .5 3 --max-iter=10" "--max-iter=10 --xlim .5 3" "--xlim .5 3 --ylim 3e-5 .2"
        ;;
    "MSA_SPA")
        exc "bsc 1200_3_6_ldpc SPA MSA comp_dec --max-iter=10" "BSC"
        exc "biawgn 1200_3_6_ldpc SPA MSA comp_dec --max-iter=10 --xlim .5 2.75" "BIAWGN"
        ;;
    "ALL")
        ARR=("BEC" "BSC_MSA" "BIAWGN_MSA" "BSC_SPA" "BIAWGN_SPA" "MSA_SPA")
        list ARR[@]
        ;;
    *)
        log "Non-existent CASE=${CASE}!"
        exit -1
        ;;
esac

log "Done!"
