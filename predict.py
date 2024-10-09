from src import *
import optuna
import argparse 
import re
import os

parser = argparse.ArgumentParser(description="Run inference on triplets")


parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="The file for the predictions"
)
parser.add_argument(
    "--cuda",
    type=int,
    required=True,
)

args= parser.parse_args() 
setting = "drug_combination_discovery"
study_name = f"{setting}_0"  # Unique identifier of the study.
storage_name = "sqlite:///studies/{}.db".format(study_name)
study = optuna.load_study(study_name, storage_name)
p = study.best_params
study.best_params
config = {"optimizer": {"alpha":p["alpha"],
                        "ratio_onedrug":p["ratio_onedrug"],
                        "learning_rate":p["learning_rate"],
                        "factor":p["factor"],
                        "clip_norm":p["clip_norm"],
                        "batch_size":p["batch_size"]},
         "network": {"embed_dim": p["embed_dim"],
                     "hidden_dim_fusion" :p["hidden_dim_fusion"],
                      "hidden_dim_mlp" : p["hidden_dim_mlp"],
                      "use_norm_bias":p["use_norm_bias"],
                      "use_norm_slope":p["use_norm_slope"],
                      "dropout_fusion":p["dropout_fusion"],
                      "num_res":p["num_res"],
                      "dropout_res":p["dropout_res"]},
         "env": {"device":f"cuda:{args.cuda}",
                 "root": "./data/",
                "setting":setting}}

device = torch.device(config["env"]["device"])
smiles = pd.read_csv("data/SMILES_ALMANAC.csv", index_col=0)
bestR = -1
nci_lines = pd.read_csv("data/nci60toidx.csv", index_col=0)
pred_df = pd.read_csv(args.file, index_col=0)
n_drugs = len([col for col in pred_df.columns if re.match("SMILES_", col)])
cell_lines = torch.Tensor(nci_lines.loc[pred_df.loc[:, "CELL_NAME"]].to_numpy().squeeze()).to(device).long()
ds = []
for n_drug in range(1, n_drugs+1):
    is_in_almanac = pred_df.loc[:, f"SMILES_{n_drug}"].isin(smiles.loc[:, "smilesString"])
    if (~is_in_almanac).any():
        not_in_almanac = pred_df.loc[:, f"SMILES_{n_drug}"].loc[~is_in_almanac].to_numpy()
        print(RuntimeWarning(f"WARNING: {not_in_almanac} are not part of the ALMANAC study"))
    fps = FingerprintFeaturizer(fp_kwargs={"nBits":2048},R=2)(pred_df.loc[:, f"SMILES_{n_drug}"])
    ds += [(torch.stack(list(fps.values())).to(device), torch.Tensor(pred_df.loc[:, f"CONC{n_drug}"].to_numpy()).to(device).unsqueeze(-1).unsqueeze(-1))]
    

for fold in range(10):
    bestR = -1
    for model_n in os.listdir("models/"):
        if re.match(f"{setting}_{fold}", model_n):
            log = torch.load(f"models/{model_n}", map_location = "cpu")["log"]
            if log["r synergies"] > bestR:
                bestR = log["r synergies"]
                best_model = model_n
    tns = torch.load(f"models/{best_model}", map_location = "cpu")
    model = Model(embed_dim=config["network"]["embed_dim"],
                      hidden_dim_fusion=config["network"]["hidden_dim_fusion"],
                      hidden_dim_mlp = config["network"]["hidden_dim_mlp"],
                      use_norm_bias=config["network"]["use_norm_bias"],
                      use_norm_slope=config["network"]["use_norm_slope"],
                      dropout_fusion=config["network"]["dropout_fusion"],
                      num_res=config["network"]["num_res"],
                      dropout_res=config["network"]["dropout_res"])
    model.load_state_dict(tns["model"])
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        out = model(cell_lines, ds)
        preds += [out.cpu().numpy().squeeze()]
mean_p = np.mean(preds, -2)
fname = re.match("(.*)[.]csv", f"{args.file}")
pred_df.assign(prediction = mean_p).to_csv(f"{fname[1]}_prediction.csv")
            