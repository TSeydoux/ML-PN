# ParticleNet Tagger

This repository provides an implementation of the **ParticleNet architecture**, from the weaver-core project (https://github.com/hqucms/weaver-core). This code was developed to create an **ML-based tagger for the $$B_s^0 \rightarrow \tau^+ \tau^-$$ decay, where $$\tau^\pm \rightarrow 3\pi^\pm$$**, in the context of FCC-ee studies.

It includes tools to:

- Preprocess data from ROOT files
- Train the ParticleNet model
- Evaluate its performance
- Automatically generate all relevant plots

---

## Installation

### 1. Install Miniconda

If you do not already have Conda installed, download **Miniconda** and install it (choose the installation path when prompted):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Initialize Conda:
```
source ~/.bashrc
source <path_to_miniconda>/etc/profile.d/conda.sh
```


### 2. Create and activate the environment

Create a dedicated Conda environment:
```
conda create -n <path>/weaver python=3.10
conda activate <path>/weaver
```


### 3. Install Weaver

Install weaver-core:
```
pip install weaver-core
```
If you get the error "No module named 'torch' inside the weaver environment", try to install it manually:
```
conda activate <path>/weaver
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### 1. Data preprocessing

Data preparation is handled by the `data_prep.py` script.

To run the preprocessing step, execute the associated shell script:
```
./data_prep.sh
```
You might need the permission first:
```
chmod +x data_prep.sh
./data_prep.sh
```
This step reads ROOT files, preprocesses the data, rescale it, and stores it in a format suitable for training. You may want to change paths in `data_prep.py` and `data_prep.sh` first.


### 2. Training the model

Configuration of data is done in `PN.yaml` and hyperparameters of the model are set in `PN.py`.
Training parameters (e.g. learning rate, batch size, number of epochs) and paths can be configured in `train_PN.sh`. To start training, run:
```
./train_PN.sh
```

This script will:

- Train the ParticleNet model using the selected parameters
- Save model results in ROOT files for both validation and testing samples
- Produce performance plots automatically


### 3. Plot customization

The appearance and layout of the plots can be modified in `plots.py`. After making changes, regenerate the plots by running:
```
./plots_PN.sh
```
You may want to change path in `plots_PN.sh` when running it standalone.

---
