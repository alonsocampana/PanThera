import pandas as pd
import numpy as np
import zipfile
import torch
from functools import lru_cache
import os
from copy import deepcopy
from synergy.combination import MuSyC, BRAID, Zimmer
from utils import * 
from torchmetrics import AUROC
from src import *  
from torch import nn
from torchmetrics import PearsonCorrCoef
import argparse 
import optuna
import uuid 

@lru_cache(maxsize=None)
def get_data(fold, setting, root, filepath="ComboDrugGrowth_Nov2017.csv", num_workers=4):
    """Load and preprocess the ALMANAC drug combination dataset.

    Args:
        fold (int): Current cross-validation fold.
        setting (str): Type of data split (e.g., random, extrapolation, etc.).
        root (str): Path to directory containing all required CSV files.
        filepath (str): File contaning the combination therapy data with the same format as the ALMANAC

    Returns:
        tuple: 
            - ds (SynergyDatasetTensor): Main synergy dataset.
            - drug_dictionary (dict): Drug fingerprint features.
            - oc_dl (DataLoader): DataLoader for one-concentration data.
            - dr_dl (DataLoader): DataLoader for dose-response data.
    """
    # Load raw synergy data
    data = pd.read_csv(root + filepath)
    
    # Load drug chemical info with special encoding due to potential non-UTF characters
    drugs = pd.read_csv(root + "drug_chemical_info.csv", encoding="latin")
    
    # Load SMILES strings indexed by NSC ID
    smiles = pd.read_csv(root + "SMILES_ALMANAC.csv", index_col=0)
    
    # Manually correct or fill in missing SMILES strings
    smiles.loc[102] = [608210, drugs.set_index("drugName").loc["Vinorelbine"].loc["smilesString"]]
    smiles.loc[103] = [753082, drugs.query("drugName ==  'Vemurafenib'").loc[:, "smilesString"].item()]
    smiles.loc[104] = [754230, drugs.query("drugName ==  'Pralatrexate'").loc[:, "smilesString"].item()]
    smiles.loc[105] = [6396, drugs.query("drugNameOfficial == 'thiotepa'").loc[:, "smilesString"].item()]
    smiles.loc[106] = [761431, drugs.query("drugNameOfficial == 'vemurafenib'").loc[:, "smilesString"].item()]
    
    # Load one-concentration and dose-response datasets
    oneconc = pd.read_csv(root + "oneconc_processed.csv", index_col=0)
    doseresp = pd.read_csv(root + "doseresp_processed.csv", index_col=0)

    # Create a unified list of all unique cell lines found across datasets
    all_cells = set(oneconc["CELL_NAME"].unique()).union(set(doseresp["CELL_NAME"].unique())).union(set(data["CELLNAME"].unique()))
    all_cells = sorted(list(all_cells))

    # Map each cell line to a unique numeric ID
    cell_map = {j: i for i, j in enumerate(all_cells)}

    # Apply numeric encoding to all cell line references
    oneconc["CELL_NAME"] = oneconc["CELL_NAME"].map(cell_map)
    doseresp["CELL_NAME"] = doseresp["CELL_NAME"].map(cell_map)
    data["CELLNAME"] = data["CELLNAME"].map(cell_map)

    # Convert tabular one-concentration data into a PyTorch dataset
    oc_ds = SDDatasetTensor(oneconc)
    oc_dl = torch.utils.data.DataLoader(oc_ds, batch_size=512, shuffle=True, num_workers=num_workers)

    # Convert dose-response data (with required dtype casting) into grouped format
    dr_ds = SDGroupedDatasetTensor(doseresp.dropna().astype({"CELL_NAME": int, "CONCENTRATION": float, "AVERAGE_GIPRCNT": float}))
    dr_dl = torch.utils.data.DataLoader(dr_ds, batch_size=512, shuffle=True, num_workers=num_workers)

    # Create a fingerprint dictionary for all drugs based on SMILES strings
    drug_dictionary = FingerprintFeaturizer(fp_kwargs={"nBits": 2048}, R=2)(smiles["smilesString"], smiles["NSC"])

    # Initialize the synergy dataset with drug features
    ds = SynergyDatasetTensor(data=data, drug_features=drug_dictionary)
    ds.training = True

    # Split the dataset depending on the experimental setting
    if setting == "random":
        random_split(ds.drs, ds.pairs, fold)
    elif setting == "synergy_discovery":
        synergy_discovery_split(ds.drs, ds.pairs, fold)
    elif setting == "drug_combination_discovery":
        drug_combination_discovery_split(ds.drs, ds.pairs, fold)
    elif setting == "smoothing":
        smoothing_split(ds.drs, fold)
    elif setting == "interpolation":
        interpolation_split(ds.drs, ds.pairs, fold)
    elif setting == "extrapolation":
        extrapolation_split(ds.drs, ds.pairs, fold)

    return ds, drug_dictionary, oc_dl, dr_dl

def train_model(config, callback_intermediate=print):
    """
    Function used for training the model and returning the performance metrics.

    Args:
        config (dict): Configuration dictionary containing all training, model, and environment parameters.
        callback_intermediate (callable): Callback function to report intermediate results after each epoch.
    """

    # Instantiate model with configuration parameters
    model = Model(
        embed_dim=config["network"]["embed_dim"],
        hidden_dim_fusion=config["network"]["hidden_dim_fusion"],
        hidden_dim_mlp=config["network"]["hidden_dim_mlp"],
        use_norm_bias=config["network"]["use_norm_bias"],
        use_norm_slope=config["network"]["use_norm_slope"],
        dropout_fusion=config["network"]["dropout_fusion"],
        num_res=config["network"]["num_res"],
        dropout_res=config["network"]["dropout_res"]
    )

    # Select computation device (CPU or CUDA device) as defined in config
    device = torch.device(config["env"]["device"])

    # Load datasets and drug representation dictionary
    ds, drug_dictionary, oc_dl, dr_dl = get_data(
        config["env"]["fold"],
        config["env"]["setting"],
        config["env"]["root"],
        config["env"]["data_path"],
        num_workers=config["env"]["num_workers"]
    )

    # Move model to selected device
    model.to(device)

    # Initialize early stopping with patience threshold
    early_stopper = EarlyStop(max_patience=20)

    # Set up optimizer (Adam) and LR scheduler (ReduceLROnPlateau)
    optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=config["optimizer"]["factor"], 
        patience=10
    )

    # Use Mean Squared Error as loss function
    loss = nn.MSELoss()

    # DataLoader for model training with shuffle and multiprocessing
    train_dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=config["optimizer"]["batch_size"], 
        shuffle=True, 
        num_workers=config["env"]["num_workers"]
    )

    # Test DataLoader using a custom collate function (likely for nested batch structures)
    test_dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=256, 
        collate_fn=collate_nested
    )

    # Final evaluation DataLoader
    final_dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=256, 
        num_workers=config["env"]["num_workers"]
    )

    # Determine number of batches to use based on ratio for one-drug training
    num_batches = int(config["optimizer"]["ratio_onedrug"] * len(train_dl))
    
    # Initialize evaluation metrics and move to device
    auroc_syn = AUROC().to(device)  # AUROC for synergy classification
    auroc_ant = AUROC().to(device)  # AUROC for antagonism classification
    r = PearsonCorrCoef().to(device)  # Pearson correlation (likely for predictions vs. ground truth)
    r_comboscore = PearsonCorrCoef().to(device)  # Potentially misused if intended to be R-squared (⚠️ see below)


    # Store alpha hyperparameter for loss combination (e.g., multi-task loss weight)
    alpha = config["optimizer"]["alpha"]

    # Initialize best synergy correlation score
    best_synergy_r = -2  # Intentionally below valid Pearson range [-1, 1] to ensure it gets updated

    # Determine whether the task is zero-shot or not
    if config["env"]["setting"] in ["smoothing", "interpolation", "extrapolation"]:
        zero_shot = False
    else:
        zero_shot = True
    for epoch in range(800):
        # Reset metrics at the beginning of each epoch to avoid accumulation
        auroc_syn.reset()
        auroc_ant.reset()
        r.reset()
        r_comboscore.reset()
    
        # Lists to track different losses per epoch
        l_train = []
        l_train_oc = []  # Losses from the ONECONC dataset
        l_train_dr = []  # Losses from the DOSERESP dataset
        l_test = []
    
        model.train()  # Set model to training mode
    
        # --- Training on ONECONC dataset (typically fixed-dose single drug responses) ---
        for i, x in enumerate(oc_dl):
            if i >= num_batches:
                break  # Stop if maximum number of batches is reached (controls balance of training sources)
    
            optimizer.zero_grad()
    
            # Forward pass: predict from cell ID and single-drug inputs (fingerprint + concentration)
            out = model(
                x[0].long().to(device),
                [(x[1][0][0].to(device), x[1][0][1].unsqueeze(-1).to(device))]  # Ensure concentration shape is (B, 1)
            )
    
            # Compute loss against ground-truth growth inhibition
            l = loss(out.squeeze(), x[2].squeeze().to(device))
    
            # Backpropagation
            l.backward()
    
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
    
            optimizer.step()
    
            # Track per-batch loss
            l_train_oc += [l.item()]
    
        # --- Training on DOSERESP dataset (typically varying-dose single drug responses) ---
        for i, x in enumerate(dr_dl):
            if i >= num_batches:
                break
    
    
            optimizer.zero_grad()
    
            # Forward pass using cell ID and drug representation
            out = model(
                x[0].long().to(device),
                [(x[1][0][0].to(device), x[1][0][1].unsqueeze(-1).to(device))]
            )
    
            # Compute MSE loss against actual dose-response data
            l = loss(out.squeeze(), x[2].squeeze().to(device))
    
            # Backprop and optimizer step
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
            optimizer.step()
    
            # Track loss
            l_train_dr += [l.item()]
        train_dl.dataset.training = True  # Enables training mode for dataset (controls masking)
        train_dl.dataset.get_complete_matrix = False  # Avoid loading full dose-response matrices during training
        
        for i, x in enumerate(train_dl):  # Iterate over the main training dataset (combinations)
               
            optimizer.zero_grad()
        
            # Forward pass using cell-line IDs and two drugs (fingerprints + concentration grids)
            out = model(
                x[4].long().to(device),
                [(x[2].to(device), x[1][:, :, :, 0].to(device)),
                 (x[3].to(device), x[1][:, :, :, 1].to(device))]
            )
        
            if zero_shot:
                # Training loss is weighted sum of normalized point-wise MSE and synergy MSE on test positions
                l = (1 - alpha) * loss(out.squeeze(), x[0].squeeze().to(device) / 100) + \
                    alpha * loss(get_synergy(out)[x[5]], get_synergy(x[0] / 100).to(device)[x[5]])
            else:
                # Avoid computing synergies during training to prevent leakage in interpolation/extrapolation settings
                l = loss(out.squeeze(), x[0].squeeze().to(device) / 100)
        
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
            optimizer.step()
            l_train += [l.item()]
        
            # Update learning rate scheduler using average training loss
            scheduler.step(np.mean(l_train))
            
            # Switch model to evaluation mode and update dataset flags
            model.eval()
            test_dl.dataset.training = False
            test_dl.dataset.get_complete_matrix = False
            
            # First pass: calculate point-wise MSE (used in all splits)
        for i, x in enumerate(test_dl):
            x = x
            try:
                with torch.no_grad():
                    out = model(
                        x[5].long().to(device),
                        [(x[3].to(device), x[1].to_padded_tensor(np.nan).unsqueeze(1).to(device)),
                         (x[4].to(device), x[2].to_padded_tensor(np.nan).unsqueeze(1).to(device))]
                    )
            
                    pred = out.squeeze()
                    obs = x[0].to_padded_tensor(np.nan).squeeze().to(device) / 100
                    l = loss(pred[~pred.isnan()], obs[~obs.isnan()])
                    l_test += [l.item()]
            except RuntimeError:
                pass
        
        # Second pass: evaluate complete matrices (ComboScore/synergy metrics)
        final_dl.dataset.training = True
        final_dl.dataset.get_complete_matrix = True
        synergies = []
        
        for i, x in enumerate(final_dl):
            x = x
            with torch.no_grad():
                out = model(
                    x[4].long().to(device),
                    [(x[2].to(device), x[1][:, :, :, 0].to(device)),
                     (x[3].to(device), x[1][:, :, :, 1].to(device))]
                )
        
                # Compute mean squared error of synergy map vs ground truth
                synergies += [((get_synergy(out) - get_synergy(x[0] / 100).to(device)) ** 2).mean().item()]
        
            x[5].to(device) 
        
            if x[5].any():  # Proceed only if test points exist in the matrix
                syn_pool = get_synergy(x[0].to(device) / 100).mean(-1).mean(-1)  # True ComboScore (averaged)
                syn_pool_obs = get_synergy(out).mean(-1).mean(-1)  # Predicted ComboScore
        
                # Pearson between predicted and true synergy values (masked)
                r.update(get_synergy(out)[x[5]].flatten(), get_synergy(x[0].to(device) / 100)[x[5]].flatten())
        
                # AUROC: classify synergistic combinations (> 0.1 synergy)
                auroc_syn.update(syn_pool_obs[x[5]], syn_pool[x[5]] > 0.1)
        
                # AUROC: classify antagonistic combinations (< -0.1 synergy)
                auroc_ant.update(-syn_pool_obs[x[5]], syn_pool[x[5]] < -0.1)
        
                # Pearson R for ComboScore regression
                r_comboscore.update(syn_pool_obs[x[5]], syn_pool[x[5]])
        
        # Finalize AUROC scores if applicable (i.e., in zero-shot settings)
        if not zero_shot:
            auc_s = None
            auc_a = None
        else:
            auc_s = auroc_syn.compute().item()
            auc_a = auroc_ant.compute().item()
        # Logging training and validation results per epoch
        print(f"""epoch : {epoch}
                train loss_oc: {np.mean(l_train_oc)}  
                train loss_dr: {np.mean(l_train_dr)} 
                train loss: {np.mean(l_train)} 
                test loss: {np.nanmean(l_test)}
                r synergies: {r.compute().item()} 
                r comboscore: {r_comboscore.compute().item()}
                auroc synergies: {auc_s} 
                auroc antagonist: {auc_a} 
                global synergies: {np.nanmean(synergies)}""")
        
        synergy_epoch = r.compute().item()
        
        # Save the best-performing model based on synergy R
        if synergy_epoch > best_synergy_r:
            best_synergy_r = synergy_epoch
            model_weights = model.state_dict().copy()  # Deep copy to avoid side effects
            log = {
                "epoch" : epoch,
                "train loss_oc": np.mean(l_train_oc),
                "train loss_dr": np.mean(l_train_dr),
                "train loss": np.mean(l_train),
                "test loss": np.nanmean(l_test),
                "r synergies": r.compute().item(),
                "r comboscore": r_comboscore.compute().item(),
                "auroc synergies": auc_s,
                "auroc antagonist": auc_a, 
                "global synergies": np.nanmean(synergies),
            }
        
        # Optional external logging or visualization hook
        callback_intermediate(epoch, synergy_epoch)
        
        # Early stopping based on train loss plateau
        if early_stopper(np.mean(l_train)):
            break
        
    # Save best model state and training log to disk
    torch.save({
        "model": model_weights,
        "log": log
    }, f"models/{config['env']['model_name']}.pt")
    
    return best_synergy_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train synergy model")

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="The fold number (an integer)."
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="The partition strategy you want to use."
    )
    parser.add_argument(
        "--cuda",
        type=int,
        required=False,
        default = 0,
        help="The GPU device index (optional)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default = "ComboDrugGrowth_Nov2017.csv",
        help="The path to a data file used to train and test the model (optional, defaults to ALMANAC)"
    )
    parser.add_argument(
        "--hyperparameter_study",
        type=str,
        required=False,
        default = None,
        help="The name of a custom hyperparameter study (optional)."
    )

    args = parser.parse_args()
    # Normalize setting for Optuna study name (shared across related partitions)
    if args.setting in ["drug_combination_discovery", "interpolation", "extrapolation", "smoothing"]:
        setting_h = "drug_combination_discovery"
    else:
        setting_h = args.setting
    if args.hyperparameter_study is None:
        study_name = f"{setting_h}_0"
    else:
        study_name = args.hyperparameter_study
    storage_name = f"sqlite:///studies/{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    p = study.best_params

    # Build config dictionary from best hyperparameters
    config = {
        "optimizer": {
            "alpha": p["alpha"],
            "ratio_onedrug": p["ratio_onedrug"],
            "learning_rate": p["learning_rate"],
            "factor": p["factor"],
            "clip_norm": p["clip_norm"],
            "batch_size": p["batch_size"]
        },
        "network": {
            "embed_dim": p["embed_dim"],
            "hidden_dim_fusion": p["hidden_dim_fusion"],
            "hidden_dim_mlp": p["hidden_dim_mlp"],
            "use_norm_bias": p["use_norm_bias"],
            "use_norm_slope": p["use_norm_slope"],
            "dropout_fusion": p["dropout_fusion"],
            "num_res": p["num_res"],
            "dropout_res": p["dropout_res"]
        },
        "env": {
            "data_path": args.data_path,
            "device": f"cuda:{args.cuda}" if args.cuda is not None else "cpu",
            "root": "./data/",
            "model_name": f"{args.setting}_{args.fold}_{str(uuid.uuid4())}",
            "setting": args.setting,
            "fold": args.fold,
            "num_workers":4,
        }
    }

    train_model(config)
