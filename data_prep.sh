#!/bin/bash

### Script to run 'data_prep.py'

source /afs/cern.ch/work/t/thseydou/public/Miniconda/etc/profile.d/conda.sh    # Activates conda
conda activate weaver                                                          # Activates the specific conda environment
python /afs/cern.ch/work/t/thseydou/public/PN/data_prep.py                     # Runs the data_prep.py code
