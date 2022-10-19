# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# from sklearn.ensemble import RandomForestRegressor as RFR

from juharmonize import JuHarmonize
from juharmonize.utils import subset_data


import argparse
from pathlib import Path
import sys

to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)
from lib.harmonize import eval_harmonizer, train_harmonizer  # noqa
from lib import io  # noqa
from lib import ml  # noqa
from lib.logging import logger, configure_logging
# Running Parameters

configure_logging("INFO")

# parse parameters
parser = argparse.ArgumentParser(description="Attributes")

parser = io.set_argparse_params(parser, use_oos=False)
parser = ml.set_argparse_params(parser)

parser.add_argument(
    "--save_dir", type=str, default="", help="Save results path"
)

parser.add_argument(
    "--problem_type",
    type=str,
    default="binary_classification",
    help="Problem type",
)

parser.add_argument(
    "--harm_n_splits", type=int, default=10, help="Folds in Harmonization"
)
parser.add_argument(
    "--regression_points", type=int, default=100, help="Regression point"
)
parser.add_argument(
    "--no-regression_search", action='store_false', default=True,
    help="Skip regression search"
)

parser.add_argument(
    "--regression_search_tol",
    type=float,
    default=2.0,
    help="Regression tolerance",
)

parser.add_argument(
    "--harmonize_mode", type=str, default="JUHA", help="Harmonization Mode"
)
parser.add_argument(
    "--n_splits", type=int, default=3, help="Numbers of CV folds"
)
parser.add_argument(
    '--fold', type=int, default=-1,
    help='If -1, run all folds, otherwise run the specified fold'
)
parser.add_argument(
    "--random_state", type=int, default=23, help="Random State use"
)

params = parser.parse_args()

# Paths
save_dir = Path(params.save_dir)

# General
n_splits = params.n_splits
random_state = params.random_state
problem_type = params.problem_type
fold_to_do = params.fold


# Harmonizaton set up
harm_n_splits = params.harm_n_splits
harmonize_mode = params.harmonize_mode


# ######################## Data loading and preprocessing
data = io.get_MRI_data(params, problem_type, use_oos=False)
X, y, sites, covars = data  # type: ignore

# ######################## Models Set ups

pred_model, stack_model = ml.get_models(params, problem_type)

# Cross varidation Parameters
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


regression_params = None
if problem_type == "regression":
    regression_params = {
        "regression_points": params.regression_points,
        "regression_search_tol": params.regression_search_tol,
        "regression_search": params.regression_search,
    }

if harmonize_mode == "cheat":
    logger.info("Cheat mode")
    cheat_model = JuHarmonize()
    X_harm = cheat_model.fit_transform(X, y, sites, covars)

    new_var = np.sum(X_harm.std(axis=0) < X.std(axis=0))
    # TODO: fix this or people will have strokes trying to understand
    logger.info(
        f"Variance decreased for {new_var} features of {X.shape[1]}"
    )
    harmonize_mode = "none"

for i_fold, (train_index, test_index) in enumerate(kf.split(X)):
    if fold_to_do >= 0 and i_fold != fold_to_do:
        continue
    logger.info(f"CV Fold: {i_fold} ...")
    X_train, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X, sites, y, covars
    )

    X_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X, sites, y, covars
    )

    harm_model = train_harmonizer(
        harmonize_mode,
        X_train,
        y_train,
        sites_train,
        covars_train,
        pred_model,
        stack_model=stack_model,
        random_state=None,
        n_splits=10,
        regression_params=regression_params
    )

    out_fold, acc_fold = eval_harmonizer(
        harm_model, X_test, y_test, sites_test, covars_test
    )

    logger.info("================================")
    logger.info(f"\tFOLD {i_fold} - SCORE: {acc_fold}")
    logger.info("================================")

    out_fname = f"{harmonize_mode}_fold_{i_fold}_of_{n_splits}_out.csv"
    to_save = pd.DataFrame({"y_true": y_test, "y_pred": out_fold})
    to_save['harmonize_mode'] = harmonize_mode
    to_save['fold'] = i_fold
    to_save.to_csv(save_dir / out_fname, sep=';')
