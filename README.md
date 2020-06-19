Message passing based decoders for LDPC codes with Python/NumPy. Includes implementations of
* **min-sum** (MSA) and **sum-product** (SPA) algorithms using **sparse matrices** (`scipy.sparse`)
* **maximum-likedlood** (ML) and **linear-programming** (LP) decoders (only for short length codes like Hamming(7,4)) based on [Using Linear Programming to Decode Binary Linear Codes](https://people.eecs.berkeley.edu/~wainwrig/Papers/FelWaiKar05.pdf)
* **ADMM** decoder based on [Decomposition Methods for Large Scale LP Decoding](https://arxiv.org/pdf/1204.0556.pdf)

for binary erasure (**BEC**), binary symmetric (**BSC**) and binary-AWGN (**biawgn**) channels.

### Dependencies
Following Python/package versions (or higher) are required.
* `Python version 3.5.2`
* `numpy version 1.12.0`
* `scipy version 0.18.1`


## Exploring pre-computed results
See all pre-compupted results in [`codes`](data/codes), [`output`](data/output) and [`plots`](data/plots). These include different codes, simulation results and plots.
<img src="data/plots/BIAWGN_MSA_ensemble.png?raw=true" width="280" />
<img src="data/plots/BSC_SPA_compare.png?raw=true" width="280" />
<img src="data/plots/MSA_SPA_BIAWGN.png?raw=true" width="280" />
<img src="data/plots/BEC_compare.png?raw=true" width="280" />
<img src="data/plots/BEC_max_iter.png?raw=true" width="280" />
<img src="data/plots/HMG_BIAWGN.png?raw=true" width="280" />


## Starting from scratch

### Running simulations
* Simulation output written to `~/scratch/decoders` by default.
* Execute `python src/codes.py 10 1200 3 6` to generate 10 random samples from `LDPC(1200,3,6)` ensemble.
* Run simulations using `run_sims.sh {PARA} {CASE} {ARGS}` command. For example,
  * `./run_sims.sh SEQL HMG` executes all Hamming code related simulations **sequentially**.
  * `./run_sims.sh SEQL REG_ENS --data-dir=./data --console` executes some regular LDPC related simulations **sequentially** while printing logs onto console.
  * `./run_sims.sh PARA REG_ENS --data-dir=./data` executes the same **in parallel**.
  * Use the latter only on a **dedicated server** as it will take large amount of CPU.
  * See [`run_sims.sh`](../master/run_sims.sh) for other choices of `{CASE}`.

* If running the simulations on [Niagara cluster](https://docs.computecanada.ca/wiki/Niagara), need to first activate a Python environment with required packages installed.
* Execute `./run_sims.sh PARA HMG --data-dir=$SCRATCH` to test if simulations run properly.
* Run all simulations in parallel within one node:
`submitjob single -e "~/projects/decoders/run_sims.sh PARA HMG MAR REG_BAD REG_ENS IREG_ENS --console" -d time=11:59:00 -D email`


### Generating plots
* Execute `./plot_results.py HMG --ext=png --data-dir=./data --plots-dir=./plots --save --silent --error=ber` to silently save Hamming code related plots.
* Replace `HMG` with `HMG REG_ENS` to save both Hamming code and regular ensemble related plots.
* If running on the Niagara cluster execute `./plot_results.py HMG REG_ENS --data-dir=$SCRATCH --plots-dir=$SCRATCH --silent --agg` to use the proper back-end for `matplotlib`.


## Individual commands

### Simulations
* `python src/main.py bec 1200_3_6_rand_ldpc_1 SPA --codeword=1 --console --params .5 .475 .45 .425 .4 .375 .35 .325 .3`
* `python src/main.py biawgn 7_4_hamming SPA --codeword=1 --params .01 .05 .1 .5 1 2 4 6 --console`
* See [`run_sims.sh`](../master/run_sims.sh) and [`simulations.py`](../master/simulations.py) for more.

### Plots
* `python src/graph.py bec --and 1200_3_6_ldpc SPA --error ber`
* `python src/graph.py bsc --and 7_4_hamming --or_ SPA ML --error wer`
* See [`plot_results.py`](../master/plot_results.py) for more.


### Generating optimal irregular LDPC codes
First of the following plots the density evolution while the second generates the optimal irregular distribution a given check node distribution (rho).
* `python src/ldpc.py plt`
* `python src/ldpc.py irg --count=10 --len=1200 --rho=5 --rate=.5`


### Luby-Transform (LT) code
Execute the following to reproduce the Figure 50.4 (histogram for LT code with length 10000) in Information Theory, Inference, and Learning Algorithms by David J.C. MacKay. Replace <c> in the first command with 0.01, 0.03, 0.1.
* `python -u src/luby.py 10000 12000 <c> .5 250 --pool=4`
* `python src/luby_graph.py .01 .03 .1`
