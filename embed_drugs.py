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
class EmbedDrugs():
    def __init__(self, n_concs, drug_dictionary, n_cell):
        self.n_cell = n_cell
        self.n_concs = n_concs
        almanac = pd.read_csv("data/ComboDrugGrowth_Nov2017.csv")
        self.fps = torch.load("/mnt/mlshare/alonsocampana/data/all_nsc_fps.pt")
        self.log_hi = pd.read_csv("data/nscs_loghi.csv")
        self.drug_dictionary = drug_dictionary
        self.min_concs = almanac.groupby(["NSC1"])["CONC1"].agg(lambda x: x.value_counts().index[0]).to_dict()
        self.max_concs = almanac.groupby(["NSC1"])["CONC1"].agg(lambda x: x.value_counts().index[2]).to_dict()
        self.csnsc2 = np.logspace(np.log10(np.array(list(self.min_concs.values()))), np.log10(np.array(list(self.max_concs.values()))), 3)
        self.csnsc2 = np.concatenate([np.zeros([105, 1]), self.csnsc2.T], 1)
        self.fps2 = torch.cat([drug_dictionary[k].unsqueeze(0) for k in list(self.min_concs.keys())], axis=0)
        NSCS = pd.read_html("https://discover.nci.nih.gov/cellminerdata/html/NSC_LIST.html")

        table_moas = NSCS[0].set_index("NSC #")

        moas_known = table_moas.loc[table_moas.loc[:, "Mechanism of action"] != "-"].dropna()

        Moas = {'DHFR':'DHFR',
                'IMPDH':'IMPDH',
                'TUBB':'TUBB',
                'Tu-stab':'Tu-stab',
                'AlkAg':'AlkAg',
                 'A7':'A7',
                'Ho':'Ho',
                'TOP2':'TOP2',
                'TOP1':'TOP1',
                'STAT':'STAT',
                'HDAC':'HDAC',
                'AKT':'PIK3/MTOR/STK/AKT',
                'PIK3':'PIK3/MTOR/STK/AKT',
                'MTOR':'PIK3/MTOR/STK/AKT',
                'STK':'PIK3/MTOR/STK/AKT',
                'HSP90':'HSP90',
                'BRAF':'BRAF',
                'IAP':'IAP', 
                'PSM':'PSM',
                'MDM2':'MDM2',
                'ALK':'ALK',
                'BRD':'BRD',
                'EGFR':'EGFR',
                'FGFR':'FGFR',
                'MET':'MET',
                'Ds':'Ds',
                'Db':'Db',
                "SMO":"SMO",
                'Apo':'Apo'}

        is_any_moa = np.zeros(len(moas_known.loc[:, "Mechanism of action"])).astype(bool)
        for m in Moas.items():
            moa = m[0].lower()
            is_moa = moas_known.loc[:, "Mechanism of action"].str.lower().str.contains(moa)
            is_any_moa = is_any_moa | is_moa
            moas_known.loc[:, "Mechanism of action"].loc[is_moa] = m[1]
        self.drugs_to_embed = moas_known.loc[moas_known.index.isin(np.array(list(self.fps.keys())))]
    def __len__(self):
        return len(self.drugs_to_embed)
    def __getitem__(self, idx):
        NSC_drug1 = int(self.drugs_to_embed.index[idx])
        conc_1  = -4
        concs1 = np.concatenate([np.zeros([1]), np.logspace(conc_1 - 1, conc_1 - 4, 3)], 0)
        cell_id = self.n_cell
        return (self.fps[NSC_drug1],
                torch.Tensor(concs1.repeat(4))[None,:],
                torch.Tensor([cell_id]))
    

ds = EmbedDrugs(n_concs = n_concs, drug_dictionary = drug_dictionary, n_cell = 0)
batch_size = 512
inference_dl = torch.utils.data.DataLoader(ds, batch_size = batch_size, num_workers=8, shuffle=False)
tns = torch.load(f"models/{best_model}")
model = ModelEmbedding(embed_dim=config["network"]["embed_dim"],
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
fps = []
drugs = []
interactions = []
for n_i, x in enumerate(inference_dl):
    with torch.no_grad():
        fp, drug_embedding, interaction_embedding = model(x[2].long().to(device), [(x[0].to(device), x[1].to(device)),])
        fps += [fp.squeeze().cpu().detach().numpy()]
        drugs += [drug_embedding.squeeze().cpu().detach().numpy()]
        interactions += [interaction_embedding.squeeze().cpu().detach().numpy()]
fps = pd.DataFrame(np.concatenate(fps), ds.drugs_to_embed.index.to_numpy())
fps.to_csv("embeddings/fingerprints.csv")
fps = pd.DataFrame(np.concatenate(drugs), ds.drugs_to_embed.index.to_numpy())
fps.to_csv("embeddings/drugs.csv")
fps = pd.DataFrame(np.concatenate(interactions), ds.drugs_to_embed.index.to_numpy())
fps.to_csv("embeddings/embeddings.csv")