from src import *  
import optuna
import argparse 
import re
import os
import torch
import pandas as pd
import numpy as np

# -----------------------------------
# 🧪 Argument Parsing
# -----------------------------------
parser = argparse.ArgumentParser(description="Run inference on triplets")
parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="The CSV file containing SMILES and metadata for prediction"
)
parser.add_argument(
    "--cuda",
    type=int,
    required=True,
    help="The GPU device index to use"
)

parser.add_argument(
    "--root",
    type=str,
    required=False,
    default = "./",
    help="The root path"
)
args = parser.parse_args()
setting = "drug_combination_discovery"
study_name = f"{setting}_0"  
storage_name = f"sqlite:///studies/{study_name}.db"

# Load best parameters from Optuna study
study = optuna.load_study(study_name, storage_name)
p = study.best_params

# Define full config from hyperparameters
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
        "device": f"cuda:{args.cuda}",
        "root": args.root,
        "setting": setting
    }
}
root = config["env"]["root"]
device = torch.device(config["env"]["device"])
# Load external data
smiles = pd.read_csv(root + "data/SMILES_ALMANAC.csv", index_col=0)
nci_lines = pd.read_csv(root + "data/nci60toidx.csv", index_col=0)
pred_df = pd.read_csv(args.file, index_col=0)

n_drugs = len([col for col in pred_df.columns if re.match("SMILES_", col)])
cell_lines = torch.Tensor(nci_lines.loc[pred_df["CELL_NAME"]].to_numpy()).squeeze().long().to(device)

# Featurize drug inputs
ds = []
for n_drug in range(1, n_drugs + 1):
    smiles_col = f"SMILES_{n_drug}"
    is_in_almanac = pred_df[smiles_col].isin(smiles["smilesString"])
    
    if (~is_in_almanac).any():
        not_in_almanac = pred_df.loc[~is_in_almanac, smiles_col].to_numpy()
        print(RuntimeWarning(f"WARNING: {not_in_almanac} are not part of the ALMANAC study"))

    fps = FingerprintFeaturizer(fp_kwargs={"nBits": 2048}, R=2)(pred_df[smiles_col])
    fps_tensor = torch.stack(list(fps.values())).to(device)
    conc_tensor = torch.tensor(pred_df[f"CONC{n_drug}"].to_numpy()).unsqueeze(-1).unsqueeze(-1).to(device)

    ds.append((fps_tensor.float(), conc_tensor.float()))
bestR = -1
preds = []

for fold in range(10):
    bestR = -1
    best_model = None
    
    for model_n in os.listdir(root + "models/"):
        if re.match(f"{setting}_{fold}", model_n):
            log = torch.load(root + f"models/{model_n}", map_location="cpu")["log"]
            if log["r synergies"] > bestR:
                bestR = log["r synergies"]
                best_model = model_n

    if best_model is None:
        print(f"No model found for fold {fold}, skipping.")
        continue

    tns = torch.load(root + f"models/{best_model}", map_location="cpu")

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
    
    model.load_state_dict(tns["model"])
    model.eval()
    model.to(device)

    with torch.no_grad():
        output = model(cell_lines.long(), ds)
        preds.append(output.cpu().numpy().squeeze())
# Average across folds
mean_p = np.mean(preds, axis=0)

# Output results to CSV with _prediction suffix
fname = re.match(r"(.*)\.csv", args.file)
output_path = f"results/{fname[1]}_prediction.csv"
pred_df.assign(prediction=mean_p).to_csv(root + output_path)

print(f"Saved predictions to: {output_path}")
