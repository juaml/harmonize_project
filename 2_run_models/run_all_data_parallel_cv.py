# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

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
from lib.logging import logger, configure_logging  # noqa
from lib.utils import check_params

# Running Parameters

configure_logging()

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
    "--regression_search",
    action="store_true",
    default=False,
    help="Do regression search",
)

parser.add_argument(
    "--regression_search_tol",
    type=float,
    default=2.0,
    help="Regression tolerance",
)

parser.add_argument(
    "--harmonize_mode", type=str, default="pretend", help="Harmonization Mode"
)
parser.add_argument(
    "--n_splits", type=int, default=3, help="Numbers of CV folds"
)

parser.add_argument(
    "--select_k",
    type=int,
    default=-1,
    help=(
        "Numbers of features to select using SelectKBest. "
        "If -1 (default), use all features."
    ),
)

parser.add_argument(
    "--fold",
    type=int,
    default=-1,
    help="If -1, run all folds, otherwise run the specified fold",
)
parser.add_argument(
    "--random_state", type=int, default=23, help="Random State use"
)

parser.add_argument(
    "--use_disk",
    action="store_true",
    default=False,
    help="Use disk for save RF",
)

parser.add_argument(
    "--n_jobs", type=int, default=None, help="Numbers of jobs for predictor"
)

params = parser.parse_args()

check_params(params)

# Paths
save_dir = Path(params.save_dir)

save_dir.mkdir(exist_ok=True, parents=True)

logger.info(f"Saving results in {save_dir.as_posix()}")

# General
n_splits = params.n_splits
random_state = params.random_state
problem_type = params.problem_type
fold_to_do = params.fold
select_k = params.select_k

# Harmonizaton set up
harm_n_splits = params.harm_n_splits
harmonize_mode = params.harmonize_mode
n_jobs = params.n_jobs
use_disk = params.use_disk

# ######################## Models Set ups

pred_model, stack_model = ml.get_models(params, problem_type)

# Cross varidation Parameters
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


regression_params = None
if problem_type == "regression":
    regression_params = {
        "regression_points": params.regression_points,
        "regression_search_tol": params.regression_search_tol,
        "regression_search": params.regression_search,
    }


# ######################## Data loading and preprocessing
data = io.get_MRI_data(params, problem_type, use_oos=False)
X, y, sites, covars = data  # type: ignore

if select_k > 0:
    logger.info(f"Selecting {select_k} best features.")
    if select_k > X.shape[1]:
        raise ValueError(
            f"Cannot select {select_k} features from {X.shape[1]} features."
        )
    if problem_type == "regression":
        fselect = SelectKBest(score_func=f_regression, k=select_k)
    else:
        fselect = SelectKBest(score_func=f_classif, k=select_k)
    X = fselect.fit_transform(X, y)
    assert X.shape[1] == select_k

cheat = False
if harmonize_mode == "cheat":
    logger.info("Cheat mode")
    cheat_model = JuHarmonize()
    X = cheat_model.fit_transform(X, y, sites, covars)

    new_var = np.sum(X.std(axis=0) < X.std(axis=0))
    # TODO: fix this or people will have strokes trying to understand
    logger.info(f"Variance decreased for {new_var} features of {X.shape[1]}")
    harmonize_mode = "none"
    cheat = True

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
        random_state=random_state,
        n_splits=harm_n_splits,
        regression_params=regression_params,
        use_disk=use_disk,
        n_jobs=n_jobs,
    )

    out_fold, acc_fold = eval_harmonizer(
        harm_model, X_test, y_test, sites_test, covars_test
    )

    out_fold_train, acc_fold_train = eval_harmonizer(
        harm_model, X_train, y_train, sites_train, covars_train
    )

    if cheat is True:
        harmonize_mode = "cheat"

    logger.info("================================")
    logger.info(
        f"\tFOLD {i_fold} - TEST SCORE: {acc_fold} "
        f"- TRAIN SCORE: {acc_fold_train}"
    )
    logger.info("================================")

    # Save TEST results
    out_fname = f"{harmonize_mode}_fold_{i_fold}_of_{n_splits}_out.csv"
    to_save = pd.DataFrame(
        {"y_true": y_test, "y_pred": out_fold, "site": sites_test}
    )
    to_save["harmonize_mode"] = harmonize_mode
    to_save["fold"] = i_fold
    out_path = save_dir / out_fname
    logger.info(f"Saving dataframe in {out_path.as_posix()}")
    to_save.to_csv(out_path, sep=";")

    # Save TRAIN results
    out_fname = f"{harmonize_mode}_fold_{i_fold}_of_{n_splits}_train.csv"
    to_save = pd.DataFrame(
        {"y_true": y_train, "y_pred": out_fold_train, "site": sites_train}
    )
    to_save["harmonize_mode"] = harmonize_mode
    to_save["fold"] = i_fold
    out_path = save_dir / out_fname
    logger.info(f"Saving dataframe in {out_path.as_posix()}")
    to_save.to_csv(out_path, sep=";")
