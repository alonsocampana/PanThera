from src import *
import optuna
import argparse 
import re

parser = argparse.ArgumentParser(description="Run inference on triplets")


parser.add_argument(
    "--setting",
    type=str,
    required=False,
    default = "drug_combination_discovery",
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
nci_lines = pd.read_csv("data/nci60toidx.csv", index_col=0)
device = torch.device(config["env"]["device"]) 
n_concs = 4
class InferenceTriplets():
    def __init__(self, n_concs, drug_dictionary, n_cell, smoke_test=False):
        self.n_cell = n_cell
        self.n_concs = n_concs
        almanac = pd.read_csv("data/ComboDrugGrowth_Nov2017.csv")
        self.triplets = []
        self.drug_dictionary = drug_dictionary
        self.min_concs = almanac.groupby(["NSC1"])["CONC1"].agg(lambda x: x.value_counts().index[0]).to_dict()
        self.max_concs = almanac.groupby(["NSC1"])["CONC1"].agg(lambda x: x.value_counts().index[2]).to_dict()
        length_dict = 100
        for d_i in range(len(drug_dictionary.values())):
            if not length_dict:
                break
            for d_j in range(len(drug_dictionary.values())):
                if not length_dict:
                    break
                for d_k in range(len(drug_dictionary.values())):
                    if not length_dict:
                        break
                    if d_i == d_j or d_i == d_j or d_j == d_k:
                        pass
                    elif d_i > d_j or d_i > d_k or d_j > d_k:
                        pass
                    else:
                        self.triplets += [[list(drug_dictionary.keys())[d_i], list(drug_dictionary.keys())[d_j], list(drug_dictionary.keys())[d_k]]]
                        if smoke_test:
                            length_dict -= 1
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        id_triplet = idx%len(self.triplets)
        cell_id = self.n_cell
        di, dj, dk = self.triplets[id_triplet]
        triplet_cs = torch.Tensor([[c1, c2, c3] 
                                   for c1 in np.concatenate([np.zeros([1]), np.logspace(np.log10(self.min_concs[di]), np.log10(self.max_concs[di]), self.n_concs-1)])
                                   for c2 in np.concatenate([np.zeros([1]), np.logspace(np.log10(self.min_concs[dj]), np.log10(self.max_concs[dj]), self.n_concs-1)]) 
                                   for c3 in np.concatenate([np.zeros([1]), np.logspace(np.log10(self.min_concs[dk]), np.log10(self.max_concs[dk]), self.n_concs-1)])])[None, :]
        return drug_dictionary[di], triplet_cs[:,  :, 0], drug_dictionary[dj], triplet_cs[:,  :, 1], drug_dictionary[dk], triplet_cs[:,  :, 2], torch.Tensor([cell_id])
    
def expected_triplets_synergy_min(expected):
    expected = deepcopy(expected)
    A1= np.array([(expected[i, 0, 0,]  * expected[0, 1:, 1:,] ) for i in range(1, expected.shape[0])])
    A1_NEG_1 = np.array([(np.ones(expected[i, 0, 0,].shape) * expected[0, 1:, 1:,] ) for i in range(1, expected.shape[0])])
    A1_NEG_2 = np.array([(expected[i, 0, 0,]  * np.ones(expected[0, 1:, 1:,].shape)) for i in range(1, expected.shape[0])])
    A1_neg = np.minimum(A1_NEG_1, A1_NEG_2)
    A1[A1_neg < 0] = A1_neg[A1_neg < 0]
    A2 = np.array([expected[0, i, 0,]  * expected[1:, 0, 1:,]  for i in range(1, expected.shape[1])])
    A2_NEG_1 =  np.array([np.ones(expected[0, i, 0,].shape) * expected[1:, 0, 1:,]  for i in range(1, expected.shape[1])])
    A2_NEG_2 =  np.array([expected[0, i, 0,]  * np.ones(expected[1:, 0, 1:,].shape) for i in range(1, expected.shape[1])])
    A2_neg = np.minimum(A2_NEG_1, A2_NEG_2)
    A2[A2_neg < 0] = A2_neg[A2_neg < 0]
    A3 = np.array([expected[0, 0, i,]  * expected[1:, 1:, 0,]  for i in range(1, expected.shape[2])])
    A3_NEG_1 = np.array([np.ones(expected[0, 0, i,].shape) * expected[1:, 1:, 0,]  for i in range(1, expected.shape[2])])
    A3_NEG_2 = np.array([expected[0, 0, i,]  * np.ones(expected[1:, 1:, 0,].shape) for i in range(1, expected.shape[2])])
    A3_neg = np.minimum(A3_NEG_1, A3_NEG_2)
    A3[A3_neg < 0] = A3_neg[A3_neg < 0]
    expected[1:, 1:, 1:] = np.minimum(np.minimum(A1.transpose(0, 1, 2), A2.transpose(1, 0, 2)),  A3.transpose(1, 2, 0))
    return expected

def expected_triplets_synergy(expected):
    expected = deepcopy(expected)
    A1= np.array([(expected[i, 0, 0,]  * expected[0, 1:, 1:,] ) for i in range(1, expected.shape[0])])
    A1_NEG_1 = np.array([(np.ones(expected[i, 0, 0,].shape) * expected[0, 1:, 1:,] ) for i in range(1, expected.shape[0])])
    A1_NEG_2 = np.array([(expected[i, 0, 0,]  * np.ones(expected[0, 1:, 1:,].shape)) for i in range(1, expected.shape[0])])
    A1_neg = np.minimum(A1_NEG_1, A1_NEG_2)
    A1[A1_neg < 0] = A1_neg[A1_neg < 0]
    A2 = np.array([expected[0, i, 0,]  * expected[1:, 0, 1:,]  for i in range(1, expected.shape[1])])
    A2_NEG_1 =  np.array([np.ones(expected[0, i, 0,].shape) * expected[1:, 0, 1:,]  for i in range(1, expected.shape[1])])
    A2_NEG_2 =  np.array([expected[0, i, 0,]  * np.ones(expected[1:, 0, 1:,].shape) for i in range(1, expected.shape[1])])
    A2_neg = np.minimum(A2_NEG_1, A2_NEG_2)
    A2[A2_neg < 0] = A2_neg[A2_neg < 0]
    A3 = np.array([expected[0, 0, i,]  * expected[1:, 1:, 0,]  for i in range(1, expected.shape[2])])
    A3_NEG_1 = np.array([np.ones(expected[0, 0, i,].shape) * expected[1:, 1:, 0,]  for i in range(1, expected.shape[2])])
    A3_NEG_2 = np.array([expected[0, 0, i,]  * np.ones(expected[1:, 1:, 0,].shape) for i in range(1, expected.shape[2])])
    A3_neg = np.minimum(A3_NEG_1, A3_NEG_2)
    A3[A3_neg < 0] = A3_neg[A3_neg < 0]
    expected[1:, 1:, 1:] = (A1.transpose(0, 1, 2) + A2.transpose(1, 0, 2) + A3.transpose(1, 2, 0))/3
    return expected

def expected_individual_multiplicative(expected):
    expected = deepcopy(expected)
    d1 = expected[1:, 0, 0]
    d2 = expected[0, 1:, 0]
    d3 = expected[0, 0, 1:]
    G1 = np.ones(d1.shape)
    G2 = np.ones(d2.shape)
    G3 = np.ones(d3.shape)
    N_CONC1 = d1.shape[-1]
    N_CONC2 = d2.shape[-1]
    N_CONC3 = d3.shape[-1]
    d1_only = d1[:,None,None] *np.ones(d2[None,:,None].shape)*np.ones(d3[None,None,:].shape)
    d2_only = np.ones(d1[:,None,None].shape)*d2[None,:,None] *np.ones(d3[None,None,:].shape)
    d3_only = np.ones(d1[:,None,None].shape)*np.ones(d2[None,:,None].shape)*d3[None,None,:] 
    d3_only[d3_only > 1] = 1
    d2_only[d2_only > 1] = 1
    d1_only[d1_only > 1] = 1
    expected_pos = d1_only * d2_only * d3_only
    expected_neg = np.minimum(np.minimum(d3_only, d2_only), d1_only)
    expected_pos[expected_neg < 0] = expected_neg[expected_neg < 0]
    expected[1:, 1:, 1:] = expected_pos
    return expected

def measure_synergy(out, synergy_fn):
    return np.array([(synergy_fn(out[i].cpu().reshape(n_concs, n_concs, n_concs).numpy()) - out[i].cpu().reshape(n_concs, n_concs, n_concs).numpy())[1:, 1:, 1:] for i in range(len(out))])

def expected_individual_min(expected):
    expected = deepcopy(expected)
    F1_i_j_0 = expected[1:, 1:, 0,][:, :, np.newaxis] # i, j[:, :, np.newaxis]  # Shape (num_points, num_points, 1)
    F1_i_0_k = expected[1:, 0, 1:,][..., np.newaxis, :]  # Shape (num_points, 1, num_points)
    F1_0_j_k = expected[0, 1:, 1:,][np.newaxis, ...]  # Shape (1, num_points, num_points)

    # Use numpy minimum function to calculate the minimum across these slices
    expected[1:, 1:, 1:] = np.minimum(np.minimum(F1_i_j_0, F1_i_0_k), F1_0_j_k)
    return expected

for cell_idx in range(61):
    n_cell = nci_lines.iloc[cell_idx].item()
    cell_name = nci_lines.index.to_numpy()[cell_idx]
    ds = InferenceTriplets(n_concs = n_concs, drug_dictionary = drug_dictionary, n_cell = n_cell)
    batch_size = 128
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
    individual_multiplicative = []
    overall_min = []
    expected_triplets = []
    expected_triplets_min = []
    absolute_potency = []
    for n_i, x in enumerate(inference_dl):
        with torch.no_grad():
            out = model(x[6].long().to(device), [(x[0].to(device), x[1].to(device)),
                               (x[2].to(device), x[3].to(device)),
                               (x[4].to(device), x[5].to(device))])
            individual_multiplicative += [measure_synergy(out, expected_individual_multiplicative).mean(-1).mean(-1).mean(-1)]
            # expected individual multiplicative tends to give slightly larger expected viabilities, it's slightly pessimistic 
            overall_min  += [measure_synergy(out, expected_individual_min).mean(-1).mean(-1).mean(-1)]
            # expected minimum tends to give slightly smaller expected viabilities, it's slightly optimistic
            expected_triplets +=[ measure_synergy(out, expected_triplets_synergy).mean(-1).mean(-1).mean(-1)]
            # expected triplets tends to give slightly larger expected viabilities, it's slightly pessimitic
            expected_triplets_min += [measure_synergy(out, expected_triplets_synergy_min).mean(-1).mean(-1).mean(-1)]
            absolute_potency += [(min(out[i].reshape(n_concs, n_concs, n_concs)[0].clip(-1, 2).min(),
            out[i].reshape(n_concs, n_concs, n_concs)[:, 0].clip(-1, 2).min(),
            out[i].reshape(n_concs, n_concs, n_concs)[:, :, 0].clip(-1, 2).min()) - out[i].min().clip(-1, 2)).item() for i in range(len(out))]

    NSCS = np.array(ds.triplets)[:len(absolute_potency)]
    db = np.concatenate([NSCS,
                    np.array(absolute_potency)[:, None],
                   np.concatenate(overall_min)[:, None],
                   np.concatenate(expected_triplets)[:, None],
                   np.concatenate(expected_triplets_min)[:, None],
                   np.concatenate(individual_multiplicative)[:, None]], 1)

    db_df = pd.DataFrame(db)
    db_df.columns = ["NSC1", "NSC2", "NSC3", "Absolute Potency", "Minimal triplets", "average multiplicative triplets", "minimal multiplicative triplets", "multiplicative individual"]
    db_df = db_df.assign(CELL_NAME = cell_name)
    db_df.sort_values("Absolute Potency").to_csv(f"db/{n_cell}_{best_model}.csv")