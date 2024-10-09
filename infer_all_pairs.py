from src import *
import optuna
import argparse 
import re

parser = argparse.ArgumentParser(description="Run inference on triplets")


parser.add_argument(
    "--setting",
    type=str,
    required=False,
    default = "synergy_discovery",
    help="The partition strategy you want to use."
)
parser.add_argument(
    "--cuda",
    type=int,
    required=True,
)

parser.add_argument(
    "--fold",
    type=int,
    required=True,
)

args= parser.parse_args()
class Model(nn.Module):
    def __init__(self, embed_dim=256,
                 hidden_dim_fusion=1024,
                 hidden_dim_mlp=1024,
                 dropout_fusion=0.1,
                 dropout_res=0.1,
                 use_norm_bias = False,
                 use_norm_slope = False,
                 num_res=2):
        super().__init__()
        self.fusion = LatentHillFusionModule(embed_dim,
                                             hidden_dim_fusion,
                                             dropout=dropout_fusion, 
                                             use_norm_bias=use_norm_bias,
                                             use_norm_slope = use_norm_slope)
        self.embed_d = nn.Sequential(nn.LazyLinear(embed_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_fusion))
        self.embed_c = nn.Embedding(num_embeddings=70, embedding_dim=embed_dim)
        self.mlp = ResNet(embed_dim, hidden_dim_mlp, dropout_res, num_res)
    def forward(self, c, drugs):
        ds = []
        for drug,conc in drugs:
            ds += [(self.embed_d(drug), conc.flatten(-2))]
        c = self.embed_c(c).squeeze(1)
        shp = drugs[0][1].shape
        h = self.fusion(c, ds)
        return self.mlp(h).reshape(shp)
        
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()
            
study_name = f"{args.setting}_0"  # Unique identifier of the study.
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
                "setting":args.setting}}
model = Model(embed_dim=config["network"]["embed_dim"],
                  hidden_dim_fusion=config["network"]["hidden_dim_fusion"],
                  hidden_dim_mlp = config["network"]["hidden_dim_mlp"],
                  use_norm_bias=config["network"]["use_norm_bias"],
                  use_norm_slope=config["network"]["use_norm_slope"],
                  dropout_fusion=config["network"]["dropout_fusion"],
                  num_res=config["network"]["num_res"],
                  dropout_res=config["network"]["dropout_res"])

drugs = pd.read_csv("data/drug_chemical_info.csv", encoding="latin")
smiles = pd.read_csv("data/SMILES_ALMANAC.csv", index_col=0)
smiles.loc[102] = [608210, drugs.set_index("drugName").loc["Vinorelbine"].loc["smilesString"]]
smiles.loc[103] = [753082, drugs.query("drugName ==  'Vemurafenib'").loc[:, "smilesString"].item()]
smiles.loc[104] = [754230, drugs.query("drugName ==  'Pralatrexate'").loc[:, "smilesString"].item()]
smiles.loc[105] = [6396, drugs.query("drugNameOfficial == 'thiotepa'").loc[:, "smilesString"].item()]
smiles.loc[106] = [761431, drugs.query("drugNameOfficial == 'vemurafenib'").loc[:, "smilesString"].item()]
drug_dictionary = FingerprintFeaturizer(fp_kwargs={"nBits":2048},R=2)(smiles.loc[:,"smilesString"], smiles.loc[:,"NSC"])

import os
bestR = -1
for model_n in os.listdir("models/"):
    if re.match(f"{args.setting}_{args.fold}", model_n):
        log = torch.load(f"models/{model_n}")["log"]
        if log["r synergies"] > bestR:
            bestR = log["r synergies"]
            best_model = model_n
        
device = torch.device(config["env"]["device"]) 
n_concs = 4
import pandas as pd
class InferencePairs():
    def __init__(self, n_concs, drug_dictionary, n_cell):
        self.n_cell = n_cell
        self.n_concs = n_concs
        almanac = pd.read_csv("data/ComboDrugGrowth_Nov2017.csv")
        self.fps = torch.load("data/all_nsc_fps.pt")
        self.log_hi = pd.read_csv("data/nscs_loghi.csv")
        self.drug_dictionary = drug_dictionary
        self.min_concs = almanac.groupby(["NSC1"])["CONC1"].agg(lambda x: x.value_counts().index[0]).to_dict()
        self.max_concs = almanac.groupby(["NSC1"])["CONC1"].agg(lambda x: x.value_counts().index[2]).to_dict()
        self.csnsc2 = np.logspace(np.log10(np.array(list(self.min_concs.values()))), np.log10(np.array(list(self.max_concs.values()))), 3)
        self.csnsc2 = np.concatenate([np.zeros([105, 1]), self.csnsc2.T], 1)
        self.fps2 = torch.cat([drug_dictionary[k].unsqueeze(0) for k in list(self.min_concs.keys())], axis=0)
    def __len__(self):
        return len(self.log_hi) * 105
    def __getitem__(self, idx):
        id_drug1 = idx//105
        drug1 = self.log_hi.iloc[id_drug1]
        NSC_drug1 = drug1.loc["NSC"]
        conc_1  = drug1.loc["LOG_HI_CONCENTRATION"]
        concs1 = np.concatenate([np.zeros([1]), np.logspace(conc_1 - 1, conc_1 - 4, 3)], 0)
        id_drug2 = idx%105
        concs2 = self.csnsc2[id_drug2]
        cell_id = self.n_cell
        return (self.fps[NSC_drug1],
                torch.Tensor(concs1.repeat(4))[None,:],
                self.fps2[id_drug2],
                torch.Tensor(np.tile(concs2, 4))[None,:],
                torch.Tensor([cell_id]))
    

for n_cell in range(60):
    ds = InferencePairs(n_concs = n_concs, drug_dictionary = drug_dictionary, n_cell = n_cell)
    batch_size = 512
    inference_dl = torch.utils.data.DataLoader(ds, batch_size = batch_size, num_workers=8, shuffle=False)
    tns = torch.load(f"models/{best_model}")
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
    syns = []
    for n_i, x in enumerate(inference_dl):
        with torch.no_grad():
            out = model(x[4].long().to(device), [(x[0].to(device), x[1].to(device)),
                               (x[2].to(device), x[3].to(device)),]).reshape(-1, 4, 4)
            exp = get_expected(out)
            syn = out[:, 1:,1:] - exp
            pooled_synergies = (syn.mean(-1).mean(-1)).detach().cpu().numpy()
        syns += [pooled_synergies]
    syns = np.concatenate(syns, 0)
    nsc1 = inference_dl.dataset.log_hi.loc[:, "NSC"].repeat(105)[:len(syns)]
    nsc2 = np.tile(np.array(list(inference_dl.dataset.min_concs.keys())), len(inference_dl.dataset.log_hi))[:len(syns)]
    pd.DataFrame({"NSC1":nsc1, "NSC2":nsc2, "synergy":syns}).assign(CELL_NAME = pd.read_csv("data/cell_mapping.csv", index_col=1).loc[n_cell].item()).to_csv(f"db_pairs/{n_cell}_{best_model}.csv")