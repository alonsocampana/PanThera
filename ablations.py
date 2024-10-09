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
from functools import lru_cache
from src import *
from utils import *
from torch import nn
from torchmetrics import PearsonCorrCoef
import argparse 
import optuna
import uuid 

@lru_cache(maxsize=None)
def get_data(fold, setting, root):
    data = pd.read_csv(root + "ComboDrugGrowth_Nov2017.csv")
    drugs = pd.read_csv("data/drug_chemical_info.csv", encoding="latin")
    smiles = pd.read_csv("data/SMILES_ALMANAC.csv", index_col=0)
    smiles.loc[102] = [608210, drugs.set_index("drugName").loc["Vinorelbine"].loc["smilesString"]]
    smiles.loc[103] = [753082, drugs.query("drugName ==  'Vemurafenib'").loc[:, "smilesString"].item()]
    smiles.loc[104] = [754230, drugs.query("drugName ==  'Pralatrexate'").loc[:, "smilesString"].item()]
    smiles.loc[105] = [6396, drugs.query("drugNameOfficial == 'thiotepa'").loc[:, "smilesString"].item()]
    smiles.loc[106] = [761431, drugs.query("drugNameOfficial == 'vemurafenib'").loc[:, "smilesString"].item()]
    oneconc = pd.read_csv("data/oneconc_processed.csv", index_col=0)
    doseresp = pd.read_csv("data/doseresp_processed.csv", index_col=0)
    all_cells = set(oneconc.loc[:, "CELL_NAME"].unique()).union(set(doseresp.loc[:, "CELL_NAME"].unique())).union(set(data.loc[:, "CELLNAME"].unique()))
    all_cells = list(all_cells)
    all_cells.sort()
    cell_map = {j:i for i,j in enumerate(all_cells)}
    oneconc.loc[:, "CELL_NAME"] = oneconc.loc[:, "CELL_NAME"].map(cell_map)
    doseresp.loc[:, "CELL_NAME"] = doseresp.loc[:, "CELL_NAME"].map(cell_map)
    data.loc[:, "CELLNAME"] = data.loc[:, "CELLNAME"].map(cell_map)
    oc_ds = SDDatasetTensor(oneconc)
    oc_dl = torch.utils.data.DataLoader(oc_ds, batch_size = 512, shuffle = True, num_workers=16)
    dr_ds = SDGroupedDatasetTensor(doseresp.dropna().astype({"CELL_NAME":int, "CONCENTRATION":float, "AVERAGE_GIPRCNT":float }))
    dr_dl = torch.utils.data.DataLoader(dr_ds, batch_size = 512, shuffle = True, num_workers=16)
    drug_dictionary = FingerprintFeaturizer(fp_kwargs={"nBits":2048},R=2)(smiles.loc[:,"smilesString"], smiles.loc[:,"NSC"])
    ds = SynergyDatasetTensor(data=data,drug_features=drug_dictionary, )
    ds.training = True
    if setting == "random":
        random_split(ds.drs, ds.pairs,fold)
    elif setting == "synergy_discovery":
        synergy_discovery_split(ds.drs,ds.pairs,fold)
    elif setting == "drug_combination_discovery":
        drug_combination_discovery_split(ds.drs,ds.pairs,fold)
    elif setting == "smoothing":
        smoothing_split(ds.drs,fold)
    elif setting == "interpolation":
        interpolation_split(ds.drs,ds.pairs,fold)
    elif setting == "extrapolation":
        extrapolation_split(ds.drs,ds.pairs,fold)
    return ds, drug_dictionary, oc_dl, dr_dl
    
def train_model(config, callback_intermediate = print):
    if config["ablation"]["linear_head"]:
        config["network"]["num_res"] = 0
    model = Model(embed_dim=config["network"]["embed_dim"],
                  hidden_dim_fusion=config["network"]["hidden_dim_fusion"],
                  hidden_dim_mlp = config["network"]["hidden_dim_mlp"],
                  use_norm_bias=config["network"]["use_norm_bias"],
                  use_norm_slope=config["network"]["use_norm_slope"],
                  dropout_fusion=config["network"]["dropout_fusion"],
                  num_res=config["network"]["num_res"],
                  dropout_res=config["network"]["dropout_res"])
    device = torch.device(config["env"]["device"])
    ds, drug_dictionary, oc_dl, dr_dl = get_data(config["env"]["fold"], config["env"]["setting"], config["env"]["root"])
    model.to(device)
    early_stopper = EarlyStop(max_patience=20)
    optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config["optimizer"]["factor"], patience=10)
    loss = nn.MSELoss()
    train_dl = torch.utils.data.DataLoader(ds, batch_size=config["optimizer"]["batch_size"], shuffle=True, num_workers=8)
    test_dl = torch.utils.data.DataLoader(ds, batch_size=256, collate_fn=collate_nested)
    final_dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=8)
    num_batches = int(config["optimizer"]["ratio_onedrug"] * len(train_dl))
    auroc_syn = AUROC().to(device)
    auroc_ant = AUROC().to(device)
    r = PearsonCorrCoef().to(device)
    r2 = PearsonCorrCoef().to(device)
    if config["ablation"]["remove_synergy"]:
        alpha = 0
    else:
        alpha = config["optimizer"]["alpha"]
    best_synergy_r = -2
    if config["env"]["setting"] in ["smoothing", "interpolation", "extrapolation"]:
        zero_shot = False
    else:
        zero_shot = True
    for epoch in range(600):
        auroc_syn.reset()
        auroc_ant.reset()
        r.reset()
        r2.reset()
        l_train = []
        l_train_oc = []
        l_train_dr = []
        l_test = []
        model.train()
        if not config["ablation"]["remove_auxiliary"]:
            for i, x in enumerate(oc_dl):
                x = x
                optimizer.zero_grad()
                out = model(x[0].long().to(device),
                            [(x[1][0][0].to(device),x[1][0][1].unsqueeze(-1).to(device)),])      
                l = loss(out.squeeze(), x[2].squeeze().to(device))
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
                optimizer.step()
                l_train_oc += [l.item()]
                if i >= num_batches:
                    break
            for i, x in enumerate(dr_dl):
                x = x
                optimizer.zero_grad()
                out = model(x[0].long().to(device),
                            [(x[1][0][0].to(device),x[1][0][1].unsqueeze(-1).to(device)),])      
                l = loss(out.squeeze(), x[2].squeeze().to(device))
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
                optimizer.step()
                l_train_dr += [l.item()]
                if i >= num_batches:
                    break
        train_dl.dataset.training = True
        train_dl.dataset.get_complete_matrix = False
        for i, x in enumerate(train_dl):
            x = x
            optimizer.zero_grad()
            out = model(x[4].long().to(device),
                        [(x[2].to(device),x[1][:, :, :, 0].to(device)),
                         (x[3].to(device), x[1][:, :, :, 1].to(device))])
            if zero_shot:
                l = (1 - alpha)*loss(out.squeeze(), x[0].squeeze().to(device)/100) + (alpha) * loss(get_synergy(out)[x[5]], get_synergy(x[0]/100).to(device)[x[5]])
            else:
                l = loss(out.squeeze(), x[0].squeeze().to(device)/100)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
            optimizer.step()
            l_train += [l.item()]
        scheduler.step(np.mean(l_train))
        model.eval()
        test_dl.dataset.training = False
        test_dl.dataset.get_complete_matrix = False
        for i, x in enumerate(test_dl):
            x = x
            with torch.no_grad():
                out = model(x[5].long().to(device),
                            [(x[3].to(device),x[1].to_padded_tensor(np.nan).unsqueeze(1).to(device)),
                            (x[4].to(device),x[2].to_padded_tensor(np.nan).unsqueeze(1).to(device)),])
                pred = out.squeeze()
                obs = x[0].to_padded_tensor(np.nan).squeeze().to(device)/100
                l = loss(pred[~pred.isnan()], obs[~obs.isnan()])
                l_test += [l.item()]
        final_dl.dataset.training = True
        final_dl.dataset.get_complete_matrix = True
        synergies = []
        for i, x in enumerate(final_dl):
            x = x
            with torch.no_grad():
                out = model(x[4].long().to(device),
                        [(x[2].to(device),x[1][:, :, :, 0].to(device)),
                         (x[3].to(device), x[1][:, :, :, 1].to(device))])    
                synergies += [((get_synergy(out) - get_synergy(x[0]/100).to(device))**2).mean().item()]
            x[5].to(device)
            if x[5].any():
                syn_pool = get_synergy(x[0].to(device)/100).mean(-1).mean(-1)
                syn_pool_obs = get_synergy(out).mean(-1).mean(-1)
                r.update(get_synergy(out)[x[5]].flatten(), get_synergy(x[0].to(device)/100)[x[5]].flatten())
                auroc_syn.update(syn_pool_obs[x[5]], syn_pool[x[5]]>0.1)
                auroc_ant.update(-syn_pool_obs[x[5]], syn_pool[x[5]]<-0.1)
                r2.update(syn_pool_obs[x[5]], syn_pool[x[5]])
        if not zero_shot:
            auc_s = None
            auc_a = None
        else:
            auc_s = auroc_syn.compute().item()
            auc_a = auroc_ant.compute().item()
        print(f"""epoch : {epoch}
        train loss_oc: {np.mean(l_train_oc)}  
        train loss_dr: {np.mean(l_train_dr)} 
        train loss: {np.mean(l_train)} 
        test loss: {np.nanmean(l_test)}
        r synergies: {r.compute().item()} 
        r comboscore: {r2.compute().item()}
        auroc synergies: {auc_s} 
        auroc antagonist: {auc_a} 
        global synergies: {np.nanmean(synergies)}""")
        synergy_epoch = r.compute().item()
        if synergy_epoch > best_synergy_r:
            best_synergy_r = synergy_epoch
            model_weights = model.state_dict().copy()
            log = {"epoch" : epoch,
            "train loss_oc": np.mean(l_train_oc),
            "train loss_dr": np.mean(l_train_dr),
            "train loss": np.mean(l_train),
            "test loss": np.nanmean(l_test),
            "r synergies": r.compute().item(),
            "r comboscore": r2.compute().item(),
            "auroc synergies": auc_s,
            "auroc antagonist": auc_a, 
            "global synergies": np.nanmean(synergies),}
        callback_intermediate(epoch, synergy_epoch)
        if early_stopper(np.mean(l_train)):
            break
    suffix = ""
    for it in config["ablation"].items():
        if it[1]:
            suffix += f"{it[0]}_"
    torch.save({"model":model_weights,
                "log":log},
               f"ablations/{config['env']['model_name']}_{suffix}.pt")
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
        help="The gpu device"
    )
    parser.add_argument(
        "--remove_auxiliary",
        action="store_true",
        required=False,
        help="Remove auxiliary data"
    )
    parser.add_argument(
        "--linear_head",
        action="store_true",
        required=False,
        help="Remove residual connections head"
    )
    parser.add_argument(
        "--remove_synergy",
        action="store_true",
        required=False,
        help="remove synergy from loss"
    )
    
    args= parser.parse_args()
    if args.setting in ["drug_combination_discovery", "interpolation", "extrapolation", "smoothing"]:
        setting_h = "drug_combination_discovery"
    else:
        setting_h = args.setting
    study_name = f"{setting_h}_0"  # Unique identifier of the study.
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
              "ablation":{"remove_auxiliary":args.remove_auxiliary,
                          "linear_head":args.linear_head,
                           "remove_synergy":args.remove_synergy},
                            
             "env": {"device":f"cuda:{args.cuda}",
                     "root": "./data/",
                     "model_name": f"{args.setting}_{args.fold}_{str(uuid.uuid4())}",
                    "setting":args.setting,
                    "fold":args.fold}}
    train_model(config)
    

