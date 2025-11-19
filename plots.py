### Create all relevant plots to evaluate/monitor progress of the model
### Paths may need to be modified in the corresponding .sh file.



## imports
import os
import glob
import uproot
import argparse
import awkward as ak
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import roc_curve, auc





def trainingPlots(logdir, directory):
    """
        Creates loss and accuracy plots for PN, from the last events file in date.

        Inputs:
            logdir: str, path to the 'runs' folder that stores TensorBoard events.
            directory: str, path to the model directory. I/O files are read/wrote in this directory using standardized names.
    """

    event_files = glob.glob(os.path.join(logdir, "**", "events.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError("No event files found.")
    event_file = max(event_files, key=os.path.getmtime)   # Catch the last modified file

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    loss_train = ea.Scalars("Loss/train (epoch)")
    loss_val = ea.Scalars("Loss/eval (epoch)")
    accuracy_train = ea.Scalars("Acc/train (epoch)")
    accuracy_val = ea.Scalars("Acc/eval (epoch)")

    df_loss_train = pd.DataFrame(loss_train)[["step", "value"]]
    df_loss_val = pd.DataFrame(loss_val)[["step", "value"]]
    df_acc_train = pd.DataFrame(accuracy_train)[["step", "value"]]
    df_acc_val = pd.DataFrame(accuracy_val)[["step", "value"]]

    # Loss plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(df_loss_train['step'], df_loss_train['value'], label='Training Loss', color="#2166ac", linestyle=":", linewidth=1, marker='o')
    plt.plot(df_loss_val['step'], df_loss_val['value'], label='Validation Loss', color="#b2182b", linewidth=1, marker='o')
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.legend(fontsize=20)
    ax.set_xlim([df_loss_train['step'].min() - 0.5, df_loss_train['step'].max() + 0.5])
    plt.grid(alpha=0.4, which="both")
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.savefig(f"{directory}/PN_loss.pdf")

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(df_acc_train['step'], df_acc_train['value'], label='Training Accuracy',color="#2166ac", linestyle=":", linewidth=1, marker='o')
    plt.plot(df_acc_val['step'], df_acc_val['value'], label='Validation Accuracy', color='red', linewidth=1, marker='o')
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Accuracy (%)', fontsize=20)
    ax.legend(fontsize=20)
    ax.set_xlim([df_acc_train['step'].min() - 0.5, df_acc_train['step'].max() + 0.5])
    plt.grid(alpha=0.4, which="both")
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.savefig(f"{directory}/PN_accuracy.pdf")



def evaluationPlots(directory):
    """
    Creates normalised counts and efficiency curves for PN. Efficiency plots are created separately for the validation and the testing samples.
    
    Input:
        directory: str, path to the model directory. I/O files are read/wrote in this directory using standardized names.
    """

    with uproot.open(f"{directory}/outputs_test.root") as root_file:
        tree_names = root_file.keys()
        tree_name = tree_names[0]
        tree = root_file[tree_name]
        branches = ["label_sig", "score_label_sig"]
        data = tree.arrays(branches)
        labels = ak.to_numpy(data["label_sig"])
        scores = ak.to_numpy(data["score_label_sig"])

    # Separate signal and background
    sig_scores = scores[labels == 1]
    bkg_scores = scores[labels == 0]

    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        print("No valid signal or background scores. Exiting.")
        return

    # Plot normalised counts as a function of the score    
    bins_sig = int(np.sqrt(min(len(sig_scores), len(bkg_scores))))   # Number of bins, same number for signal and bkg

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(sig_scores, bins=bins_sig, density=True, color="#b2182b", histtype='step', linewidth=1.5)   # Plots outline of the signal; 'density=True' means that the area under the curve is set to 1
    ax.hist(sig_scores, bins=bins_sig, density=True, color="#b2182b", histtype='stepfilled', alpha=0.3, linewidth=1.5, label="$B_s \\to \\tau^+ \\tau^-$")   # Fills the area of the signal
    ax.hist(bkg_scores, bins=bins_sig, density=True, color="#2166ac", histtype='step', linewidth=1.5, label="Inc. $Z^0 \\to b\\bar{b}$")

    ax.set_xlabel("Score", fontsize=20)
    ax.set_ylabel("Normalised counts", fontsize=20)
    ax.legend(loc="upper center", fontsize=20)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')
    ax.get_xaxis().get_offset_text().set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.savefig(f"{directory}/PN_score.pdf")   # Set the proper file path


    # Plot efficiency as a function of PN cut in each sample
    fig, ax = plt.subplots(figsize=(12, 8))

    PN_cuts = np.linspace(0.05, 5.05, 500)
    
    N_sig = len(sig_scores)
    N_bkg = len(bkg_scores)

    cut_vals = []
    eff_sig = []
    eff_bkg = []
    for x in PN_cuts:
        cut_val = float(x)
        cut_vals.append(cut_val)
        cut_val = 1 - pow(10, -cut_val)
        eff_sig.append((sig_scores > cut_val).sum() / N_sig)   # Prepare to plot signal if the score is above the cut
        eff_bkg.append((bkg_scores > cut_val).sum() / N_bkg)   # Same for background
    
    # Create the plot
    plt.plot(cut_vals, eff_sig, color="#b2182b", label="$B_s \\to \\tau^+ \\tau^-$", linewidth=2)
    plt.plot(cut_vals, eff_bkg, color="#2166ac", linestyle=":", label="Inc. $Z^0 \\to b\\bar{b}$", linewidth=2)

    plt.xlabel("1 - Score", fontsize=20)
    plt.ylabel("Efficiency", fontsize=20)
    plt.legend(loc="lower left", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(0, 4.1)
    plt.xticks([0, 1, 2, 3, 4, 5], ["$10^0$", "$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "$10^{-4}$", "$10^{-5}$"])   # Might want to change the number of ticks to fit to the data
    plt.yscale('log')
    ymin,ymax = plt.ylim()
    plt.ylim(10e-6, 2)   # Might want to change the first limit to fit to the data
    plt.grid(alpha=0.4, which="both")
    fig.savefig(f"{directory}/PN_efficiency_test.pdf")   # Set the proper file path

    with uproot.open(f"{directory}/outputs_val.root") as root_file:
        tree_names = root_file.keys()
        tree_name = tree_names[0]
        tree = root_file[tree_name]
        branches = ["label_sig", "score_label_sig"]
        data = tree.arrays(branches)
        labels = ak.to_numpy(data["label_sig"])
        scores = ak.to_numpy(data["score_label_sig"])

    # Plot efficiency as a function of PN cut in each sample
    fig, ax = plt.subplots(figsize=(12, 8))

    PN_cuts = np.linspace(0.05, 5.05, 500)
    
    sig_scores = scores[labels == 1]
    bkg_scores = scores[labels == 0]

    N_sig = len(sig_scores)
    N_bkg = len(bkg_scores)
    if N_sig == 0 or N_bkg == 0:
        print("No valid signal or background scores. Exiting.")
        return

    cut_vals = []
    eff_sig = []
    eff_bkg = []
    for x in PN_cuts:
        cut_val = float(x)
        cut_vals.append(cut_val)
        cut_val = 1 - pow(10, -cut_val)
        eff_sig.append((sig_scores > cut_val).sum() / N_sig)   # Prepare to plot signal if the score is above the cut
        eff_bkg.append((bkg_scores > cut_val).sum() / N_bkg)   # Same for background
    
    # Create the plot
    plt.plot(cut_vals, eff_sig, color="#b2182b", label="$B_s \\to \\tau^+ \\tau^-$", linewidth=2)
    plt.plot(cut_vals, eff_bkg, color="#2166ac", linestyle=":", label="Inc. $Z^0 \\to b\\bar{b}$", linewidth=2)

    plt.xlabel("1 - Score", fontsize=20)
    plt.ylabel("Efficiency", fontsize=20)
    plt.legend(loc="lower left", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(0, 4.1)
    plt.xticks([0, 1, 2, 3, 4, 5], ["$10^0$", "$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "$10^{-4}$", "$10^{-4}$"])   # Might want to change the number of ticks to fit to the data
    plt.yscale('log')
    ymin,ymax = plt.ylim()
    plt.ylim(10e-6, 2)   # Might want to change the first limit to fit to the data
    plt.grid(alpha=0.4, which="both")
    fig.savefig(f"{directory}/PN_efficiency_val.pdf")   # Set the proper file path



def ROC(directory):
    """
        Creates a ROC curve for PN. Plots are based on the outputs.root file. Plot is created for the testing samples.

        Input: 
            directory: str, path to the model directory. I/O files are read/wrote in this directory using standardized names.
    """

    with uproot.open(f"{directory}/outputs_test.root") as root_file:
        tree_names = root_file.keys()
        tree_name = tree_names[0]
        tree = root_file[tree_name]
        branches = ["label_sig", "score_label_sig"]
        data = tree.arrays(branches)
        labels = ak.to_numpy(data["label_sig"])
        scores = ak.to_numpy(data["score_label_sig"])

    assert len(scores) == len(labels), "Scores and labels length mismatch"   # Sanity check

    # Calculate FPR, TPR, and AUC
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(fpr, tpr, lw=1.5, color="#2166ac", label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0., 1.], [0., 1.], linestyle="--", color="gray", label='50/50')   # Plot the baseline for a random classifier

    # Set limits and labels
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.xlabel('False Positive Rate', fontsize=20)   # Background efficiency
    plt.ylabel('True Positive Rate', fontsize=20)   # Signal efficiency
    plt.legend(loc="lower right", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.grid(True)
    
    fig.savefig(f"{directory}/PN_ROC.pdf")   # Save the figure





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Path to training logs")   # Modify paths in the corresponding .sh if needed
    parser.add_argument("--directory", required=True, help="Path to model directory")
    args = parser.parse_args()

    trainingPlots(logdir=args.logdir, directory=args.directory)
    evaluationPlots(directory=args.directory)
    ROC(directory=args.directory)