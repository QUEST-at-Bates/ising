#!/bin/bash
#SBATCH --job-name=ising2D_test    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cberger@smith.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --partition=phyq              # Which partition to use
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --output=ising_2D_%j.log   # Standard output 
#SBATCH --error=err_ising2D_%j.log   # Standard output and error log

pwd; hostname; date

echo "Running 2D Ising C++ script on a single CPU core"

./ising

date
