from train_synergy import *
import argparse
import optuna
import uuid

def train_model_optuna(trial, config):
    """Wrapper function that produces trials using Optuna
    trial: the instance optuna passes to each trial
    config: The configuration object"""
    def optuna_step_callback(epoch, train_r):
        """
        A callback that checks for early stopping at the end of each epoch
        """
        trial.report(train_r, step = epoch) # Report the R(synergy) at each epoch
        if np.isnan(train_r): 
            raise optuna.TrialPruned()
        if trial.should_prune():
            raise optuna.TrialPruned()
    # Hyperparameters grid used during hyperaprameter search
    config["network"] = { "embed_dim": trial.suggest_int("embed_dim", 64, 1024),
                    "hidden_dim_fusion": trial.suggest_int("hidden_dim_fusion", 64, 2048),
                    "hidden_dim_mlp": trial.suggest_int("hidden_dim_mlp", 64, 2048),
                    "use_norm_bias": trial.suggest_categorical("use_norm_bias", [True, False]),
                    "use_norm_slope": trial.suggest_categorical("use_norm_slope", [True, False]),
                    "dropout_fusion": trial.suggest_float("dropout_fusion", 0.0, 0.5),
                    "num_res": trial.suggest_int("num_res", 1, 6),
                    "dropout_res": trial.suggest_float("dropout_res", 0.0, 0.5)}
    config["optimizer"] = {"alpha": trial.suggest_float("alpha", 0.0, 1.0),
                                    "ratio_onedrug": trial.suggest_float("ratio_onedrug", 0.0, 1.0),
                                    "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
                                    "factor": trial.suggest_float("factor", 0.1, 0.75),
                                    "clip_norm": trial.suggest_int("clip_norm", 0.1, 20),
                                    "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512])}
    try:
        return train_model(config, optuna_step_callback)
    except Exception as e: # Error handling, so the hyperparameter optimization does not stop in case a configuration is not valid
        print(e)
        return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters study")
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="The partition strategy you want to use."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default = "ComboDrugGrowth_Nov2017.csv",
        help="The path to a data file used to train and test the model (optional, defaults to ALMANAC)"
    )
    
    parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="The CUDA device to use (an integer)."
    )   
    args= parser.parse_args()
    setting = args.setting
    cuda_device = args.cuda
    config = {"env":{"root": "./data/",
                "data_path":args.data_path,
                "setting":setting,
                "fold":0,
                "model_name": f"{args.setting}_0hp_{str(uuid.uuid4())}",
                "device": f"cuda:{cuda_device}",
                 "num_workers":4,}}
    objective = lambda x: train_model_optuna(x, config)
    study_name = f"{setting}_new"
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                direction='maximize',
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=30,
                                                               n_warmup_steps=10,
                                                               interval_steps=10))
    study.optimize(objective, n_trials=100)
