#!/bin/bash
#SBATCH --job-name="My test job"
#SBATCH --time=00:01:00
#SBATCH --mem=1M
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH -o %j.out
#SBATCH -e %j.err

# executable
sleep 50s