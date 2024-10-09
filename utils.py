import pandas as pd
import numpy as np
import zipfile
import torch
from functools import lru_cache
import os

def return_top(srs):
    concs,counts = np.unique(srs, return_counts = True)
    top_counts = counts.argsort().argsort()
    return (concs[top_counts <=4], )
class SynergyDataset():
    def __init__(self, root = "./data/", training = True):
        self.training = training
        self.root = root
        data = pd.read_csv(root + "ComboDrugGrowth_Nov2017.csv")
        missing_concs = data.loc[:,"NSC2"].isna()
        na_concs = data.loc[missing_concs]
        self.sorted_idx_drugs = na_concs.set_index(["NSC1", "CELLNAME"]).loc[:, ["CONC1", "PERCENTGROWTH"]].sort_index()
        concs_per_pair = data.dropna().groupby(["NSC1", "NSC2", "CELLNAME"])["CONC1", "CONC2"].agg(lambda x: len(return_top(x)[0]))
        self.grouped = data.dropna().groupby(['NSC1', 'NSC2', 'CELLNAME', "CONC1", "CONC2"])["PERCENTGROWTH", "EXPECTEDGROWTH"].median().drop("EXPECTEDGROWTH", axis=1)
        indexed = self.grouped.reset_index().set_index(["NSC1", "NSC2", "CELLNAME"]).sort_index()
        self.pairs = indexed.index.drop_duplicates().to_numpy()
        if os.path.exists(root+"ALMANACdrs.pt"):
            self._load_drs()
        else:
            self._preprocess()
            self._save_drs()
    def _save_drs(self):
        torch.save(self.drs, self.root+"ALMANACdrs.pt")
    def _load_drs(self):
        self.drs = torch.load(self.root+"ALMANACdrs.pt")
    def _preprocess(self):
        self.drs = []
        for pair in self.pairs:
            d1, d2, c  = pair
            mt1 = self.grouped.loc[d1].loc[d2].loc[c].reset_index()
            d1_cs = mt1.loc[:, "CONC1"].unique()
            d2_cs = mt1.loc[:, "CONC2"].unique()
            data_d1 = self.sorted_idx_drugs.loc[d1, c].reset_index(drop=True).set_index("CONC1").loc[d1_cs].reset_index().groupby("CONC1")["PERCENTGROWTH"].median()
            data_d2 = self.sorted_idx_drugs.loc[d2, c].reset_index(drop=True).set_index("CONC1").loc[d2_cs].reset_index().groupby("CONC1")["PERCENTGROWTH"].median()
            mt2 = data_d1.reset_index().assign(CONC2=0)
            mt3 = data_d2.reset_index()
            mt3 = mt3.assign(CONC2=mt3.loc[:, "CONC1"].to_numpy(), CONC1=0)
            mt = pd.concat([mt1,
            mt2,
            mt3], axis=0)
            self.drs += [DRMatrix(mt)]
    def __len__(self):
        return len(self.pairs)
    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        d1, d2, c = self.pairs[idx]
        return self.drs[idx]
class ComboScoreDataset():
    def __init__(self, data, drug_dictionary):
        self.data = data
        self.drug_dictionary = drug_dictionary
    def __len__(self):
        return len(self.data)
    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        d1 = self.drug_dictionary[instance.loc["NSC1"]]
        d2 = self.drug_dictionary[instance.loc["NSC2"]]
        c = torch.Tensor([int(instance.loc["CELLNAME"])])
        y = torch.Tensor([instance.loc["ComboScore"]]) /100
        return d1, d2, c, y


    
class DRMatrix():
    def __init__(self, data, drug1= None, drug2=None):
        self.data = data
        self.matrix = self.data.pivot(columns = "CONC2", index="CONC1", values = "PERCENTGROWTH")
        self.matrix.loc[0, 0] = 100
        self.train_mask = torch.ones(self.matrix.shape).bool()
        self.row_concs = self.matrix.index.to_numpy()
        self.col_concs = self.matrix.columns.to_numpy()
        self.y = torch.Tensor(self.matrix.to_numpy())
        self.z = self._get_tensors_concs()
    def _get_tensors_concs(self):
        c1s = torch.Tensor(self.row_concs)
        c2s = torch.Tensor(self.col_concs)
        return torch.cat([c1s.unsqueeze(-1).repeat(1, c2s.size(0)).unsqueeze(-1), c2s.unsqueeze(-2).repeat(c1s.size(0), 1).unsqueeze(-1)], -1)
    def _set_train_test_mask(self, new_mask):
         self.train_mask = new_mask
    def _get_train(self, pad = "zero"):
        y_, z_ = self.y.clone(), self.z.clone()
        if pad == "zero":
            y_[~self.train_mask] = 100
            z_[:, :, 0][~self.train_mask] = 0
            z_[:, :, 1][~self.train_mask] = 0
        else:
            raise NotImplementedError
        return y_, z_
    def _get_test(self):
        y_, z_ = self.y.clone(), self.z.clone()
        return y_[~self.train_mask], z_[~self.train_mask]
def get_expected(tns): # Used in ALMANAC
    """
    https://wiki.nci.nih.gov/download/attachments/338237347/ALMANAC%20Data%20Fields.docx?version=1&modificationDate=1513948677000&api=v2
    """
    d1 = tns[:, 0, 1:]
    d2 = tns[:, 1:, 0]
    G1 = d1.new_ones(d1.shape)
    G2 = d2.new_ones(d2.shape)
    N_CONC1 = d1.size(-1)
    N_CONC2 = d2.size(-1)
    d1_ = d1.unsqueeze(-2).repeat(1, N_CONC2, 1)
    d2_ = d2.unsqueeze(-1).repeat(1, 1, N_CONC1)
    d_min = torch.minimum(d1_, d2_)
    exp = (torch.minimum(d1, G1).unsqueeze(-2) * torch.minimum(d2, G2).unsqueeze(-1))
    exp[d_min < 0] = d_min[d_min < 0]
    return exp

def get_expected_np(tns): # Used in ALMANAC
    """
    https://wiki.nci.nih.gov/download/attachments/338237347/ALMANAC%20Data%20Fields.docx?version=1&modificationDate=1513948677000&api=v2
    """
    d1 = tns[:, 0, 1:]
    d2 = tns[:, 1:, 0]
    G1 = np.ones(d1.shape)
    G2 = np.ones(d2.shape)
    N_CONC1 = d1.shape[-1]
    N_CONC2 = d2.shape[-1]
    # Equivalent of d1.unsqueeze(-2).repeat(1, N_CONC2, 1)
    d1_ = np.expand_dims(d1, axis=-2)
    d1_ = np.tile(d1_, (1, N_CONC2, 1))
    # Equivalent of d2.unsqueeze(-1).repeat(1, 1, N_CONC1)
    d2_ = np.expand_dims(d2, axis=-1)
    d2_ = np.tile(d2_, (1, 1, N_CONC1))
    # Minimum operation
    d_min = np.minimum(d1_, d2_)
    # Equivalent of (torch.minimum(d1, G1).unsqueeze(-2) * torch.minimum(d2, G2).unsqueeze(-1))
    exp = np.expand_dims(np.minimum(d1, G1), axis=-2) * np.expand_dims(np.minimum(d2, G2), axis=-1)
    # Assign values where d_min < 0
    exp[d_min < 0] = d_min[d_min < 0]
    return exp

def get_expected_HSAM(tns):
    """
    https://arxiv.org/pdf/2210.00802.pdf
    """
    d1 = tns[:, 0, 1:]
    d2 = tns[:, 1:, 0]
    N_CONC1 = d1.size(-1)
    N_CONC2 = d2.size(-1)
    d1_ = d1.unsqueeze(-2).repeat(1, N_CONC2, 1)
    d2_ = d2.unsqueeze(-1).repeat(1, 1, N_CONC1)
    d_min = torch.minimum(d1_, d2_)
    return d_min
def get_synergy( tns, how = "ALMANAC"):
    if how == "ALMANAC":
        get_expected_fn = get_expected
    elif how == "HSAM":
        get_expected_fn = get_expected_HSAM
    syn =  get_expected_fn(tns) - (tns[:, 1:, 1:])
    return syn

def interpolation_split(drs,  triplets,fold, n_folds=10, seed=3558, hold_out=-2):
    np.random.seed(seed)
    all_drugs = np.unique([pair[0] for pair in  triplets] + [pair[1] for pair in triplets])
    np.random.shuffle(all_drugs)
    folds = np.array_split(all_drugs, n_folds)
    test_fold = folds[fold]
    for i, dr in enumerate(drs):
        triplet = triplets[i]
        if triplet[0] in test_fold:
            dr.train_mask[-2, :] = False
        if triplet[1] in test_fold:
            dr.train_mask[:, -2] = False

def extrapolation_split(drs,  triplets,fold, n_folds=10, seed=3558, hold_out=-1):
    np.random.seed(seed)
    all_drugs = np.unique([pair[0] for pair in triplets] + [pair[1] for pair in triplets])
    np.random.shuffle(all_drugs)
    folds = np.array_split(all_drugs, n_folds)
    test_fold = folds[fold]
    for i, dr in enumerate(drs):
        triplet = triplets[i]
        if triplet[0] in test_fold:
            dr.train_mask[-1, :] = False
        if triplet[1] in test_fold:
            dr.train_mask[:, -1] = False
            
def smoothing_split(drs, fold, n_folds=None, seed=3558, synergy_size = (3, 3)):
    np.random.seed(seed)
    n_cols, n_rows = synergy_size
    x = np.repeat(np.arange(1, n_rows+1)[None, :], n_rows, axis=0).flatten()
    y = np.repeat(np.arange(1, n_cols+1)[:, None], n_cols, axis=1).flatten()
    idxs = []
    for i in range(len(drs)):
        idx = np.vstack([x, y]).T
        np.random.shuffle(idx)
        idxs += [idx]
    for n_d, dr in enumerate(drs):
        idx = idxs[n_d][fold]
        dr.train_mask[idx[0], idx[1]] = False
        
def synergy_discovery_split(drs,  triplets,fold, n_folds=25, seed=3558):
    np.random.seed(seed)
    all_drugs = np.unique([pair[0] for pair in triplets] + [pair[1] for pair in triplets])
    np.random.shuffle(all_drugs)
    folds = np.array_split(all_drugs, n_folds)
    test_fold = folds[fold]
    for i, dr in enumerate(drs):
        triplet = triplets[i]
        if triplet[0] in test_fold:
            dr.train_mask[1:, 1:] = False
        if triplet[1] in test_fold:
            dr.train_mask[1:, 1:] = False
    return test_fold
            
def drug_combination_discovery_split(drs,  triplets,fold, n_folds=25, seed=3558):
    np.random.seed(seed)
    all_combs = np.unique([str(pair[0])+str(pair[1]) for pair in triplets])
    np.random.shuffle(all_combs)
    folds = np.array_split(all_combs, n_folds)
    test_fold = folds[fold]
    for i, dr in enumerate(drs):
        triplet = triplets[i]
        concat1 = str(triplet[0]) + str(triplet[1])
        concat2 = str(triplet[1]) + str(triplet[0])
        if (concat1 in test_fold) or (concat2 in test_fold):
            dr.train_mask[1:, 1:] = False
    return test_fold
def random_split(drs,  triplets,fold, n_folds=10, seed=3558):
    np.random.seed(seed)
    idxs = np.arange(len(drs))
    np.random.shuffle(idxs)
    folds = np.array_split(idxs, n_folds)
    fold = folds[fold]
    for i in fold:
        drs[i].train_mask[1:, 1:] = False
        
import rdkit
from rdkit.Chem import AllChem
class FingerprintFeaturizer():
    def __init__(self,
                 fingerprint = "morgan",
                 R=2, 
                 fp_kwargs = {},
                 transform = torch.Tensor):
        """
        Get a fingerprint from a list of molecules.
        Available fingerprints: MACCS, morgan, topological_torsion
        R is only used for morgan fingerprint.
        fp_kwards passes the arguments to the rdkit fingerprint functions:
        GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint, GetTopologicalTorsionFingerprint
        """
        self.R = R
        self.fp_kwargs = fp_kwargs
        self.fingerprint = fingerprint
        if fingerprint == "morgan":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x, self.R, **fp_kwargs)
        elif fingerprint == "MACCS":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(x, **fp_kwargs)
        elif fingerprint == "topological_torsion":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(x, **fp_kwargs)
        self.transform = transform
    def __call__(self, smiles_list, drugs = None):
        drug_dict = {}
        if drugs is None:
            drugs = np.arange(len(smiles_list))
        for i in range(len(smiles_list)):
            try:
                smiles = smiles_list[i]
                molecule = AllChem.MolFromSmiles(smiles)
                feature_list = self.f(molecule)
                f = np.array(feature_list)
                if self.transform is not None:
                    f = self.transform(f)
                drug_dict[drugs[i]] = f
            except:
                drug_dict[drugs[i]] = None
        return drug_dict
    def __str__(self):
        """
        returns a description of the featurization
        """
        return f"{self.fingerprint}Fingerprint_R{self.R}_{str(self.fp_kwargs)}"
    
class SDDatasetTensor():
    def __init__(self, data, root = "./data/"):
        self.data = data
        self.drug_fs = torch.load(root + "all_nsc_fps.pt")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        inst = self.data.iloc[idx]
        drug = self.drug_fs[inst.loc["NSC"]]
        cell = torch.Tensor([inst.loc["CELL_NAME"]])
        z = torch.Tensor([inst.loc["CONCENTRATION"]])
        y = torch.Tensor([inst.loc["AVERAGE_GIPRCNT"]/100])
        return cell, [(drug, z), ], y

class SDGroupedDatasetTensor():
    def __init__(self, data, root = "./data/"):
        self.data = data
        self.drug_fs = torch.load(root + "all_nsc_fps.pt")
        indexed = data.set_index(["NSC", "CELL_NAME"]).sort_index()
        self.pairs = indexed.index.unique()
        self.data = indexed
    def __len__(self):
        return len(self.pairs)
    @lru_cache(maxsize=None)
    def __getpair__(self, pair):
        return self.data.loc[pair].pivot(columns = "CONCENTRATION")/100
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        inst = self.__getpair__(pair)
        NSC, C = pair      
        drug = self.drug_fs[NSC]
        cell = torch.Tensor([C])
        z = inst.droplevel(-2, 1).columns.to_numpy()
        y = inst.to_numpy().squeeze(0)
        dif_len =  5 - len(z)
        if dif_len > 0:
            padding_idx = np.random.choice(np.arange(len(z)), dif_len)
            z = np.concatenate([z, z[padding_idx]])
            y = np.concatenate([y, y[padding_idx]])
        z = torch.Tensor(z)
        y = torch.Tensor(y)
        return cell, [(drug, z), ], y
    
class SynergyDatasetTensor():
    def __init__(self, data, root = "./data/", training = True, drug_features=None, line_features=None, get_complete_matrix = False):
        self.training = training
        self.root = root
        self.drug_features = drug_features
        self.line_features = line_features
        self.get_complete_matrix = get_complete_matrix
        data = data
        if drug_features is None:
            self.drug_features = {d:torch.Tensor([i]) for i, d in enumerate(data.loc[:, "NSC1"].unique())}
        else:
            self.drug_features = drug_features
        if line_features is None:
            self.line_features = {d:torch.Tensor([d]) for i, d in enumerate(data.loc[:, "CELLNAME"].unique())}
        else:
            self.line_features = line_features
        missing_concs = data.loc[:,"NSC2"].isna()
        na_concs = data.loc[missing_concs]
        self.sorted_idx_drugs = na_concs.set_index(["NSC1", "CELLNAME"]).loc[:, ["CONC1", "PERCENTGROWTH"]].sort_index()
        concs_per_pair = data.dropna().groupby(["NSC1", "NSC2", "CELLNAME"])["CONC1", "CONC2"].agg(lambda x: len(return_top(x)[0]))
        self.grouped = data.dropna().groupby(['NSC1', 'NSC2', 'CELLNAME', "CONC1", "CONC2"])["PERCENTGROWTH", "EXPECTEDGROWTH"].median().drop("EXPECTEDGROWTH", axis=1)
        indexed = self.grouped.reset_index().set_index(["NSC1", "NSC2", "CELLNAME"]).sort_index()
        self.pairs = indexed.index.drop_duplicates().to_numpy()
        if os.path.exists(root+"ALMANACdrs.pt"):
            self._load_drs()
        else:
            self._preprocess()
            self._save_drs()
    def _save_drs(self):
        torch.save(self.drs, self.root+"ALMANACdrs.pt")
    def training(self, training):
        self.training = training
    def _load_drs(self):
        self.drs = torch.load(self.root+"ALMANACdrs.pt")
    def _preprocess(self):
        self.drs = []
        for pair in self.pairs:
            d1, d2, c  = pair
            mt1 = self.grouped.loc[d1].loc[d2].loc[c].reset_index()
            d1_cs = mt1.loc[:, "CONC1"].unique()
            d2_cs = mt1.loc[:, "CONC2"].unique()
            data_d1 = self.sorted_idx_drugs.loc[d1, c].reset_index(drop=True).set_index("CONC1").loc[d1_cs].reset_index().groupby("CONC1")["PERCENTGROWTH"].median()
            data_d2 = self.sorted_idx_drugs.loc[d2, c].reset_index(drop=True).set_index("CONC1").loc[d2_cs].reset_index().groupby("CONC1")["PERCENTGROWTH"].median()
            mt2 = data_d1.reset_index().assign(CONC2=0)
            mt3 = data_d2.reset_index()
            mt3 = mt3.assign(CONC2=mt3.loc[:, "CONC1"].to_numpy(), CONC1=0)
            mt = pd.concat([mt1,
            mt2,
            mt3], axis=0)
            self.drs += [DRMatrix(mt)]
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        d1, d2, c = self.pairs[idx]
        d1 = self.drug_features[int(d1)]
        d2 = self.drug_features[int(d2)]
        c = self.line_features[c]        
        if self.training:
            y, z = self.drs[idx]._get_train()
        else:
            y, z = self.drs[idx]._get_test()
        if self.get_complete_matrix:
            y, z = self.drs[idx].y, self.drs[idx].z
        
        return (y, z,
            d1,
            d2,
            c,
                (~self.drs[idx].train_mask).any())
    
    
def collate_nested(batch):
    y = []
    c1 = []
    c2 = []
    d1 = []
    d2 = []
    c = []
    t = []
    for el in batch:
        y += [el[0]]
        c1 += [el[1][..., 0]]
        c2 += [el[1][..., 1]]
        d1 += [el[2]]
        d2 += [el[3]]
        c += [el[4]]
        t += [el[5]]
    return torch.nested.as_nested_tensor(y), torch.nested.as_nested_tensor(c1), torch.nested.as_nested_tensor(c2), torch.stack(d1, 0), torch.stack(d2, 0), torch.stack(c, 0), torch.stack(t, 0)

class EarlyStop():
    def __init__(self, max_patience, maximize=False):
        self.maximize=maximize
        self.max_patience = max_patience
        self.best_loss = None
        self.patience = max_patience + 0
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        elif loss < self.best_loss:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        else:
            self.patience -= 1
        return not bool(self.patience)