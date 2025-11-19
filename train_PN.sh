#!/bin/bash
source /afs/cern.ch/work/t/thseydou/public/Miniconda/etc/profile.d/conda.sh
conda activate weaver

# Exit on error and print commands
set -e
set -x

# === Variables to change ===
number_of_epoch=50

# === Define paths ===
DATA_DIR="/afs/cern.ch/work/t/thseydou/public/PN/preprocessed_data"
MODEL_DIR="/afs/cern.ch/work/t/thseydou/public/PN/trained_models/PN_${number_of_epoch}"
YAML_CONFIG="/afs/cern.ch/work/t/thseydou/public/PN/PN.yaml"
MODEL_SCRIPT="/afs/cern.ch/work/t/thseydou/public/PN/PN.py"
LOG_DIR="/afs/cern.ch/work/t/thseydou/public/PN/runs"

# Loop to create 3 models:
for MODEL_IDX in 1 2 3; do
    MODEL_DIR="/afs/cern.ch/work/t/thseydou/public/PN/trained_models/PN_${NUMBER_OF_EPOCH}_${MODEL_IDX}"
    mkdir -p "${MODEL_DIR}"

    # === Number of samples per epoch (training and validation) ===
    SAMPLES_PER_EPOCH=227611
    SAMPLES_PER_EPOCH_VAL=48773

    # === Weaver command ===
    time weaver \
        --data-train "${DATA_DIR}/train.root" \
        --data-val "${DATA_DIR}/val.root" \
        --data-test "${DATA_DIR}/test.root" \
        --data-config "${YAML_CONFIG}" \
        --network-config "${MODEL_SCRIPT}" \
        --model-prefix "${MODEL_DIR}/nets" \
        --num-workers 1 \
        --in-memory \
        --fetch-step 1 \
        --batch-size 512 \
        --start-lr 1e-2 \
        --num-epochs ${number_of_epoch} \
        --samples-per-epoch ${SAMPLES_PER_EPOCH} \
        --samples-per-epoch-val ${SAMPLES_PER_EPOCH_VAL} \
        --optimizer ranger \
        --log "${MODEL_DIR}/PN.log" \
        --tensorboard "${MODEL_DIR}/tensorboard"

    time weaver \
        --predict \
        --data-test "${DATA_DIR}/val.root" \
        --data-config "${YAML_CONFIG}" \
        --network-config "${MODEL_SCRIPT}" \
        --model-prefix "${MODEL_DIR}/nets" \
        --predict-output "${MODEL_DIR}/outputs_val.root"

    time weaver \
        --predict \
        --data-test "${DATA_DIR}/test.root" \
        --data-config "${YAML_CONFIG}" \
        --network-config "${MODEL_SCRIPT}" \
        --model-prefix "${MODEL_DIR}/nets" \
        --predict-output "${MODEL_DIR}/outputs_test.root"

    # === Create plots ===
    python /afs/cern.ch/work/t/thseydou/public/PN/plots.py \
        --logdir "${LOG_DIR}" \
        --directory "${MODEL_DIR}"
done