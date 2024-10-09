import pandas as pd
import numpy as np
import zipfile
import torch
from functools import lru_cache
import os
from copy import deepcopy
from synergy.combination import MuSyC, BRAID, Zimmer
from utils import *
import argparse
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import gc




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sigmoid baseline")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model you want to use (either MuSyC, BRAID or Zimmer"
    )
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
        "--cpus",
        type=int,
        required=False,
        default = 16,
        help="Number of cpus used for MP"
    )

    parser.add_argument(
        "--scale",
        action = "store_true",
        required=False,
    )

    args= parser.parse_args()
    ds = SynergyDataset()
    def process_dr(dr):
        with torch.no_grad():
            predictions = []
            observations = []
            syn_obs = []
            syn_pred = []
            norm_mat = dr.matrix.to_numpy()[None,:,:]/100
            min_e = norm_mat.min().item()
            max_e = norm_mat.max().item()
            exp = get_expected_np(norm_mat).squeeze()
            test_idx = np.array([[(j, i) for i in range(dr.matrix.shape[0])] for j in range(dr.matrix.shape[1])])[~dr.train_mask.numpy()]
            val = [dr.matrix.iloc[id[0], id[1]].item()  for id in test_idx]
            c2s = [dr.matrix.index[id[0]].item()  for id in test_idx]
            c1s = [dr.matrix.columns[id[1]].item()  for id in test_idx]
            test = deepcopy(dr.matrix[~dr.train_mask.numpy()])
            drm = dr.matrix.copy(deep=True)
            drm[~dr.train_mask.numpy()] = np.nan
            df = drm.stack().reset_index().dropna()
            trg = df[0]/100
            if args.scale:
                trg = ((trg - min_e)/(max_e - min_e))
            if args.model == "MuSyC":
                model = MuSyC()
            elif args.model == "BRAID":
                model = BRAID()
            elif args.model == "Zimmer":
                model = Zimmer()
            model.fit(df['CONC1'].to_numpy().squeeze(), df['CONC2'].to_numpy().squeeze(), trg.to_numpy().squeeze())
            for i in range(len(val)):
                pred = ((model.E(c1s[i], c2s[i])))
                #undo the minmax scaling
                if args.scale:
                    pred = (pred*(max_e - min_e)) + min_e
                    pred = pred
                obs = (val[i])/100
                predictions += [pred]
                observations += [obs]
                test_ = test_idx[i]
                if (test_ == 0).any():
                    s_obs = np.nan
                    s_pred = np.nan
                else:
                    test_ -= 1
                    s_obs = exp[test_[0], test_[1]].item() - obs
                    s_pred = exp[test_[0], test_[1]].item() - pred
                syn_obs += [s_obs]
                syn_pred += [s_pred]
            return [predictions, observations, syn_obs, syn_pred]
    def preprocess_dr(dr):
        norm_mat = dr.matrix.to_numpy()[None,:,:]/100
        min_e = norm_mat.min().item()
        max_e = norm_mat.max().item()
        exp = get_expected_np(norm_mat).squeeze()
        test_idx = np.array([[(j, i) for i in range(dr.matrix.shape[0])] for j in range(dr.matrix.shape[1])])[~dr.train_mask.numpy()]
        val = [dr.matrix.iloc[id[0], id[1]].item()  for id in test_idx]
        c2s = [dr.matrix.index[id[0]].item()  for id in test_idx]
        c1s = [dr.matrix.columns[id[1]].item()  for id in test_idx]
        test = deepcopy(dr.matrix[~dr.train_mask.numpy()])
        drm = dr.matrix.copy(deep=True)
        drm[~dr.train_mask.numpy()] = np.nan
        df = drm.stack().reset_index().dropna()
        trg = df[0]/100
        if args.scale:
            trg = (trg - min_e)/(max_e - min_e)
        return [df, trg, min_e, max_e, exp, test_idx, val, c2s, c1s, test]
    def process_dr_p(inputs):
        df, trg, min_e, max_e, exp, test_idx, val, c2s, c1s, test = inputs 
        predictions = []
        observations = []
        syn_obs = []
        syn_pred = []
        if args.scale:
            trg = (trg - min_e)/(max_e - min_e)
        if args.model == "MuSyC":
            model = MuSyC()
        elif args.model == "BRAID":
            model = BRAID()
        elif args.model == "Zimmer":
            model = Zimmer()
        model.fit(df['CONC1'].to_numpy().squeeze(), df['CONC2'].to_numpy().squeeze(), trg.to_numpy().squeeze())
        for i in range(len(val)):
            pred = ((model.E(c1s[i], c2s[i])))
            #undo the minmax scaling
            if args.scale:
                pred = (pred*(max_e - min_e)) + min_e
            obs = (val[i])/100
            predictions += [pred]
            observations += [obs]
            test_ = test_idx[i]
            if (test_ == 0).any():
                s_obs = np.nan
                s_pred = np.nan
            else:
                test_ -= 1
                s_obs = exp[test_[0], test_[1]].item() - obs
                s_pred = exp[test_[0], test_[1]].item() - pred
            syn_obs += [s_obs]
            syn_pred += [s_pred]
        return [predictions, observations, syn_obs, syn_pred]
    if args.setting == "smoothing":
        smoothing_split(ds.drs, args.fold)
    if args.setting == "interpolation":
        interpolation_split(ds.drs,ds.pairs, args.fold)
    if args.setting == "extrapolation":
        extrapolation_split(ds.drs,ds.pairs, args.fold)
    pr_dr = [preprocess_dr(dr) for dr in ds.drs]
    with Pool(args.cpus) as pool:
        out = pool.map(process_dr_p, pr_dr)
    try:
        out = np.array(out)
        predictions = out[:, 0]
        observations = out[:, 1]
        syn_obs = out[:, 2]
        syn_pred = out[:, 3]
    except ValueError:
        predictions = []
        observations = []
        syn_obs = []
        syn_pred = []
        for o in out:
            predictions += o[0]
            observations += o[1]
            syn_obs += o[2]
            syn_pred += o[3]
        predictions = np.array(predictions)
        observations = np.array(observations)
        syn_obs = np.array(syn_obs)
        syn_pred = np.array(syn_pred)
    is_na = np.isnan(predictions)
    is_syn_na = np.isnan(syn_pred)
    pd.DataFrame.from_dict({"pred": predictions, "obs": observations, "syn_obs": syn_obs, "syn_pred": syn_pred}).to_csv(f"results/ALMANAC_{args.model}_{args.setting}_{args.fold}_dump.csv")
    pd.Series({"model": args.model,
               "fold":args.fold,
           "nans":np.isnan(predictions).sum(),
           "MSE":mean_squared_error(predictions[~is_na], observations[~is_na]),
           "R_synergy": pearsonr(syn_obs[~is_syn_na], syn_pred[~is_syn_na])[0],
           "R": pearsonr(predictions[~is_na], observations[~is_na])[0],
           "R2":r2_score(predictions[~is_na],
                         observations[~is_na])}).to_csv(f"results/ALMANAC_{args.model}_{args.setting}_{args.fold}.csv")
    