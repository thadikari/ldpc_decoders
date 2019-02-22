#!/usr/bin/env bash

#local:
#./plot_results.sh BEC ../temp ../plots png --silent
#./plot_results.sh ALL ../temp ../plots png --silent

#niagara:
#~/projects/decoders/src/plot_results.sh ALL $SCRATCH $SCRATCH png "--silent --agg"


CASE=${1}
DATA_DIR=${2}
PLOTS_DIR=${3}
EXT=${4:-png}
OTHER=${5}
DIRS="--data-dir=$DATA_DIR --plots-dir=$PLOTS_DIR"

log () { echo "plot|$CASE|$1"; }
run () { echo ">> $1"; eval "$1"; }

plot_1 () {
    local CHANNEL=$1
    local DECODER=$2
    local ARGS1=$3
    local ARGS2=$4
    local ARGS3=$5

    run "python stats.py $CHANNEL 1200_3_6_rand_ldpc $DECODER --data-dir=$DATA_DIR"
    run "python graph.py $CHANNEL 1200_3_6_rand_ldpc $DECODER ensemble $DIRS $ARGS1 --save ${CASE}_ensemble.$EXT $OTHER"
    run "python graph.py $CHANNEL 1200_3_6_rand_ldpc $DECODER compare $DIRS $ARGS2 --extra 1200_3_6_ldpc --save ${CASE}_compare.$EXT $OTHER"
    run "python graph.py $CHANNEL 1200_3_6_ldpc $DECODER max_iter $DIRS $ARGS3 --save ${CASE}_max_iter.$EXT $OTHER"
}

case ${CASE} in
    "BEC")
        plot_1 bec SPA "--xlim .3 .5 --ylim 2e-7 .5" "--max-iter=10 --xlim .3 .5 --ylim 3e-5 .5" ""
        ;;
    "BSC_MSA")
        plot_1 bsc MSA "--xlim 0.02 0.08 --ylim 6e-6 .2" "--max-iter=10 --xlim 0.015 0.08" ""
        ;;
    "BIAWGN_MSA")
        plot_1 biawgn MSA "--xlim .25 2.75 --ylim 1e-5 .2" "--max-iter=10 --xlim .25 2.75" "--xlim .25 2.75"
        ;;
    "BSC_SPA")
        plot_1 bsc SPA "" "--max-iter=100" ""
        ;;
    "BIAWGN_SPA")
        plot_1 biawgn SPA "--xlim .5 2.25" "--max-iter=100 --xlim .5 2.5" ""
        ;;
    "MSA_SPA")
        run "python graph.py bsc 1200_3_6_ldpc SPA MSA comp_dec $DIRS --max-iter=10 --save bsc_MSA_vs_SPA.$EXT $OTHER"
        run "python graph.py biawgn 1200_3_6_ldpc SPA MSA comp_dec $DIRS --xlim .5 2.75 --save biawgn_MSA_vs_SPA.$EXT $OTHER"
        ;;
    "ALL")
        dqt='"'
        ARR=("BEC" "BSC_MSA" "BIAWGN_MSA" "BSC_SPA" "BIAWGN_SPA" "MSA_SPA")
        for it in ${ARR[@]}; do run "$0 ${it} ${DATA_DIR} ${PLOTS_DIR} ${EXT} ${dqt}${OTHER}${dqt}"; done
        ;;
    *)
        log "Non-existent CASE=${CASE}!"
        exit -1
        ;;
esac

log "Done!"
