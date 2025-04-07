#!/bin/bash

#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --job-name=myjob
#SBATCH --qos=orchid
#SBATCH --time=20:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"


### Provide path to yout python/conda virtual enviornment
### or you can try
source /home/users/amorgan/Calving_Fronts_and_Where_to_Find_Them/.venv/bin/activate
PROGRAM="train_the_model.py"

python ${PROGRAM}

BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"