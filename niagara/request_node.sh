#!/bin/bash
#SBATCH --account=def-sdraper
#SBATCH --ntasks-per-node=80
#SBATCH --nodes=1

#SBATCH --time=12:00:00

echo "Sleeping at: `date`."
sleep 99d  # sleep for 99 days
echo "Finished at: `date`."
