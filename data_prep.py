### Script that reads branches in ROOT files and reconstruct data for training with the appropriate rescaling.
### It plots all the branches used for training (input variables) to check that the rescaling is right.
### You may want to change paths.



import os
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt





# Paths to the data files and output
path_sig = '/eos/experiment/fcc/ee/analyses_storage/flavor/Bs2TauTau/flatNtuples/winter2023/analysis_stage1_withSimpleCut/p8_ee_Zbb_ecm91_EvtGen_Bs2TauTauTAUHADNU/chunk_0.root'   # We only use chunk_0 because we want to have ~50/50% of signal/background in the training sample, so chunk_0 is already enough
path_bkg = '/eos/experiment/fcc/ee/analyses_storage/flavor/Bs2TauTau/flatNtuples/winter2023/analysis_stage1_withSimpleCut/p8_ee_Zbb_ecm91'
path_out = '/afs/cern.ch/work/t/thseydou/public/PN/preprocessed_data'



# Branches to read from the ROOT files
branches = ["TauCand1_x", "TauCand1_y", "TauCand1_z", "TauCand1_chi2", "TauCand1_thrustangle", "TauCand1_q",
            "TauCand2_x", "TauCand2_y", "TauCand2_z", "TauCand2_chi2", "TauCand2_thrustangle", "TauCand2_q",
            "TauCand1_eta", "TauCand1_phi", 
            "TauCand2_eta", "TanCand2_phi",
            "TauCand1_pion1px", "TauCand1_pion1py", "TauCand1_pion1pz", "TauCand1_pion1q", "TauCand1_pion1d0", "TauCand1_pion1z0",
            "TauCand1_pion2px", "TauCand1_pion2py", "TauCand1_pion2pz", "TauCand1_pion2q", "TauCand1_pion2d0", "TauCand1_pion2z0",
            "TauCand1_pion3px", "TauCand1_pion3py", "TauCand1_pion3pz", "TauCand1_pion3q", "TauCand1_pion3d0", "TauCand1_pion3z0",
            "TauCand2_pion1px", "TauCand2_pion1py", "TauCand2_pion1pz", "TauCand2_pion1q", "TauCand2_pion1d0", "TauCand2_pion1z0",
            "TauCand2_pion2px", "TauCand2_pion2py", "TauCand2_pion2pz", "TauCand2_pion2q", "TauCand2_pion2d0", "TauCand2_pion2z0",
            "TauCand2_pion3px", "TauCand2_pion3py", "TauCand2_pion3pz", "TauCand2_pion3q", "TauCand2_pion3d0", "TauCand2_pion3z0"]

# Final variables to keep for training
save_vars = ["tau_eta", "tau_phi", "tau_x", "tau_y", "tau_z", "tau_q", "tau_chi2", "tau_thrustangle",
            "pion_eta", "pion_phi", "pion_px", "pion_py", "pion_pz", "pion_q", "pion_d0", "pion_z0",
            "label_bkg", "label_sig"]



# Load the TTrees
print("Loading signal TTree...")
signal = uproot.open(f"{path_sig}:events").arrays(branches, library="ak")
signal["label_bkg"] = ak.Array(np.zeros(len(signal), dtype="int8"))  # label_bkg = 0 for signal
signal["label_sig"] = ak.Array(np.ones(len(signal), dtype="int8"))   # label_sig = 1 for signal

print("Loading background TTree...")
bkg_arrays = []
for i in range(0,100):
    if i == 51:
        continue   # Skip corrupted file
    bkg_file = f'{path_bkg}/chunk_{i}.root'
    if os.path.exists(bkg_file):
        try:
            arr = uproot.open(bkg_file)["events"].arrays(branches, library="ak")
            arr["label_bkg"] = ak.Array(np.ones(len(arr), dtype="int8"))   # label_bkg = 1 for background
            arr["label_sig"] = ak.Array(np.zeros(len(arr), dtype="int8"))   # label_sig = 0 for background
            bkg_arrays.append(arr)
        except Exception as e:
            print(f"Skipping {bkg_file} due to read error: {e}")
background = ak.concatenate(bkg_arrays, axis=0)
del bkg_arrays  # Free memory

# Merge and shuffle
print("Merging and shuffling...")
combined = ak.concatenate([signal, background], axis=0)
index = np.random.permutation(len(combined))
combined = combined[index]

# Compute energies, etas, and phis for pions
print("Computing kinematic quantities...")
pion_m = 0.13957061   # GeV (from the PDG)
for tau in [1, 2]:
    for i in [1, 2, 3]: 
        px = combined[f'TauCand{tau}_pion{i}px']
        py = combined[f'TauCand{tau}_pion{i}py']
        pz = combined[f'TauCand{tau}_pion{i}pz']
        p = np.sqrt(px**2 + py**2 + pz**2)
        eta = 0.5 * np.log((p + pz) / (p - pz + 1e-8))   # Safety: avoid division by zero with 1e-8
        phi = np.arctan2(py, px)
        combined[f'TauCand{tau}_pion{i}eta'] = eta
        combined[f'TauCand{tau}_pion{i}phi'] = phi

# Construct data in the expected size, with rescaling
combined["tau_eta"] = ak.Array(np.stack([
    combined["TauCand1_eta"],
    combined["TauCand2_eta"],
], axis=1))

combined["tau_phi"] = ak.Array(np.stack([
    combined["TauCand1_phi"],
    combined["TauCand2_phi"],
], axis=1))

combined["tau_x"] = ak.Array(np.stack([
    combined["TauCand1_x"] / 2.0,   # Rescaling shows better results when applied directly here than in the .yaml
    combined["TauCand2_x"] / 2.0,
], axis=1))

combined["tau_y"] = ak.Array(np.stack([
    combined["TauCand1_y"] / 2.0,
    combined["TauCand2_y"] / 2.0,
], axis=1))

combined["tau_z"] = ak.Array(np.stack([
    combined["TauCand1_z"] / 2.0,
    combined["TauCand2_z"] / 2.0,
], axis=1))

combined["tau_q"] = ak.Array(np.stack([
    combined["TauCand1_q"],
    combined["TauCand2_q"],
], axis=1))

combined["tau_chi2"] = ak.Array(np.stack([
    (combined["TauCand1_chi2"] / 350.0),
    (combined["TauCand2_chi2"] / 2500.0),
], axis=1))

combined["tau_thrustangle"] = ak.Array(np.stack([
    combined["TauCand1_thrustangle"],   # Rescaling this variable leads to worse results
    combined["TauCand2_thrustangle"],
], axis=1))

combined["pion_eta"] = ak.Array(np.stack([
    combined["TauCand1_pion1eta"],
    combined["TauCand1_pion2eta"],
    combined["TauCand1_pion3eta"],
    combined["TauCand2_pion1eta"],
    combined["TauCand2_pion2eta"],
    combined["TauCand2_pion3eta"],
], axis=1))

combined["pion_phi"] = ak.Array(np.stack([
    combined["TauCand1_pion1phi"],
    combined["TauCand1_pion2phi"],
    combined["TauCand1_pion3phi"],
    combined["TauCand2_pion1phi"],
    combined["TauCand2_pion2phi"],
    combined["TauCand2_pion3phi"],  
], axis=1))

combined["pion_px"] = ak.Array(np.stack([
    combined["TauCand1_pion1px"] / 2.0,
    combined["TauCand1_pion2px"] / 2.0,
    combined["TauCand1_pion3px"] / 2.0,
    combined["TauCand2_pion1px"] / 2.0,
    combined["TauCand2_pion2px"] / 2.0,
    combined["TauCand2_pion3px"] / 2.0,
], axis=1))

combined["pion_py"] = ak.Array(np.stack([
    combined["TauCand1_pion1py"] / 2.0,
    combined["TauCand1_pion2py"] / 2.0,
    combined["TauCand1_pion3py"] / 2.0,
    combined["TauCand2_pion1py"] / 2.0,
    combined["TauCand2_pion2py"] / 2.0,
    combined["TauCand2_pion3py"] / 2.0,
], axis=1))

combined["pion_pz"] = ak.Array(np.stack([
    combined["TauCand1_pion1pz"] / 2.0,
    combined["TauCand1_pion2pz"] / 2.0,
    combined["TauCand1_pion3pz"] / 2.0,
    combined["TauCand2_pion1pz"] / 2.0,
    combined["TauCand2_pion2pz"] / 2.0,
    combined["TauCand2_pion3pz"] / 2.0,
], axis=1))

combined["pion_q"] = ak.Array(np.stack([
    combined["TauCand1_pion1q"],
    combined["TauCand1_pion2q"],
    combined["TauCand1_pion3q"],
    combined["TauCand2_pion1q"],
    combined["TauCand2_pion2q"],
    combined["TauCand2_pion3q"],
], axis=1))

combined["pion_d0"] = ak.Array(np.stack([
    combined["TauCand1_pion1d0"],
    combined["TauCand1_pion2d0"],
    combined["TauCand1_pion3d0"],
    combined["TauCand2_pion1d0"],
    combined["TauCand2_pion2d0"],
    combined["TauCand2_pion3d0"],
], axis=1))

combined["pion_z0"] = ak.Array(np.stack([
    combined["TauCand1_pion1z0"],
    combined["TauCand1_pion2z0"],
    combined["TauCand1_pion3z0"],
    combined["TauCand2_pion1z0"],
    combined["TauCand2_pion2z0"],
    combined["TauCand2_pion3z0"],
], axis=1))

# Plot input variables
for var in save_vars:
    arr = combined[var]
    if ak.num(arr, axis=1) is None: continue   # Don't plot labels
    n_components = ak.num(arr[0])

    plt.figure()
    for i in range(n_components):   # Plot all variables inside a branch on the same plot
        comp_data = ak.to_numpy(arr[:, i])
        plt.hist(comp_data, bins=50, alpha=0.5)
    plt.title(var)
    plt.xlabel("Value")
    plt.ylabel("Counts")
    plt.grid(True)
    os.makedirs(f"{path_out}/plots_inputs", exist_ok=True)   # Creates a file 'plots_inputs' in 'path_out' if it doesn't exist already
    plt.savefig(f"{path_out}/plots_inputs/{var}_dist.pdf")
    plt.close()

# Split into train/val/test sets
print("Splitting into train/val/test sets...")
n = len(combined)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

train = combined[:n_train]
val = combined[n_train:n_train+n_val]
test = combined[n_train+n_val:]

print("Number of events per datasets:")
print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

# Write output files
print("Writing output files...")
os.makedirs(path_out, exist_ok=True)   # Create output directory if it doesn't exist

with uproot.recreate(f"{path_out}/train.root") as f:
    f["events"] = {var: np.array(train[var]) for var in save_vars}
with uproot.recreate(f"{path_out}/val.root") as f:
    f["events"] = {var: np.array(val[var]) for var in save_vars}
with uproot.recreate(f"{path_out}/test.root") as f:
    f["events"] = {var: np.array(test[var]) for var in save_vars}

print("Data preparation complete. All files created:")
print(f"  {path_out}/train.root")
print(f"  {path_out}/val.root")
print(f"  {path_out}/test.root")
