#!/bin/bash
source /afs/cern.ch/work/t/thseydou/public/Miniconda/etc/profile.d/conda.sh    # Activates conda
conda activate weaver                                                          # Activates the specific conda environment

LOG_DIR="/afs/cern.ch/work/t/thseydou/public/PN/runs"                          # Path to the 'runs' folder. Modify as needed
MODEL_DIR="/afs/cern.ch/work/t/thseydou/public/PN/trained_models/PN_50"        # Path to the model directory. Modify as needed

python /afs/cern.ch/work/t/thseydou/public/PN/plots.py \
    --logdir "${LOG_DIR}" \
    --directory "${MODEL_DIR}"                                                 # Runs the Pyhton script