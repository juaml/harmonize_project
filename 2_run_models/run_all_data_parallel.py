#%%
from warnings import WarningMessage
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

# from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
)
from juharmonize import (
    JuHarmonize,
    JuHarmonizeRegressor,
    JuHarmonizeClassifier,
    JuHarmonizePredictor,
)
from juharmonize.utils import subset_data
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import VALID_METRICS
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing
import termplotlib as tpl
from skrvm import RVR, RVC

import argparse
from pathlib import Path
import sys


sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from lib.utils import eval_harmonizer, show_hist, train_harmonizer  # noqa

# Running Parameters

_valid_models = {
    "binary_classification": ["gssvm", "rvc"],
    "regression": ["gsgpr", "gssvm", "ridgecv", "rvr"],
}

# parse parameters
parser = argparse.ArgumentParser(description="Attributes")
parser.add_argument("--data_dir", type=str, default="", help="Data store path")
parser.add_argument(
    "--save_dir", type=str, default="", help="Save results path"
)
parser.add_argument(
    "--n_high_var_feats", type=int, default=100, help="High variance features"
)
parser.add_argument(
    "--sites_use", nargs="+", type=str, default=["all"], help="Used sites"
)
parser.add_argument(
    "--sites_oos",
    nargs="+",
    type=str,
    default=None,
    help="Out of distribution sites",
)
parser.add_argument(
    "--unify_sites",
    nargs="+",
    type=str,
    default=["IXI", "CORR", "AOMIC"],
    help="Sites to unify",
)
parser.add_argument(
    "--problem_type",
    type=str,
    default="binary_classification",
    help="Problem type",
)
parser.add_argument(
    "--random_sites", type=bool, default=False, help="If randomized sites"
)
parser.add_argument(
    "--pca", type=bool, action="store_true", default=False, help="Use PCA"
)
parser.add_argument(
    "--scaler",
    type=bool,
    action="store_true",
    default=False,
    help="Use scaler (Robust if --pca is false, otherwise Standard)",
)
parser.add_argument(
    "--covars", type=str, default=None, help="If randomized sites"
)
parser.add_argument(
    "--pred_model", type=str, default="svm", help="Prediction model to use"
)
parser.add_argument(
    "--stack_model", type=str, default="rf", help="Stacked model to use"
)
parser.add_argument(
    "--harm_n_splits", type=int, default=10, help="Folds in Harmonization"
)
parser.add_argument(
    "--harm_regression_points", type=int, default=100, help="Regression point"
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
    help='If -1, run all folds, otherwise run the specified fold')'
)
parser.add_argument(
    "--random_state", type=int, default=23, help="Random State use"
)

params = parser.parse_args()


def unify_sites(sites, unify_sites):

    # make all names uppercase
    unify_sites = [element.upper() for element in unify_sites]

    # Transform to DF to use isin function
    unify_sites = pd.DataFrame(unify_sites)

    # Unify the IXI sites
    if unify_sites.isin(["IXI"]).any()[0]:

        sites.replace(
            to_replace={"IXI/Guys": "IXI", "IXI/HH": "IXI", "IXI/IOP": "IXI"},
            inplace=True,
        )

    # Unify the AOMIC sites
    if unify_sites.isin(["AOMIC"]).any()[0]:

        sites.replace(
            to_replace={"ID1000": "AOMIC", "PIOP1": "AOMIC", "PIOP2": "AOMIC"},
            inplace=True,
        )

    # Unify the CORR sites
    if unify_sites.isin(["CORR"]).any()[0]:
        sites_use_CoRR = [
            "BMB_1",
            "BNU_1",
            "BNU_2",
            "BNU_3",
            "HNU_1",
            "IACAS",
            "IBATRT",
            "IPCAS_1",
            "IPCAS_2",
            "IPCAS_3",
            "IPCAS_4",
            "IPCAS_5",
            "IPCAS_6",
            "IPCAS_7",
            "IPCAS_8",
            "JHNU_1",
            "LMU_1",
            "LMU_2",
            "LMU_3",
            "MPG_1",
            "MRN_1",
            "NKI_1",
            "NYU_1",
            "NYU_2",
            "SWU_1",
            "SWU_2",
            "SWU_3",
            "SWU_4",
            "UM",
            "UPSM_1",
            "UWM",
            "Utah_1",
            "Utah_2",
            "XHCUMS",
        ]

        for site_CoRR in sites_use_CoRR:
            sites.replace(to_replace={site_CoRR: "CoRR"}, inplace=True)

    return sites


# Paths
data_dir = Path(params.data_dir)
save_dir = Path(params.save_dir)

# Data variables
covars = params.covars
n_high_var_feats = params.n_high_var_feats

# Site variables
sites_oos = params.sites_oos
sites_use = params.sites_use

# General
n_splits = params.n_splits
random_state = params.random_state
problem_type = params.problem_type
fold_to_do = params.fold

# Regression parameters
pred_model = params.pred_model
stack_model = params.stack_model
pca = params.pca
scaler = params.scaler

# Harmonizaton set up
harm_n_splits = params.harm_n_splits
harmonize_mode = params.harmonize_mode
regression_search_tol = params.regression_search_tol
harm_regression_points = params.harm_regression_points

# ######################## Data loading and preprocessing

X_df = pd.read_csv(data_dir / "final_data" / "X_final.csv", header=0)
X_df.reset_index(inplace=True)
Y_df = pd.read_csv(data_dir / "final_data" / "Y_final.csv", header=0)
Y_df.reset_index(inplace=True)

# ############## Format data
# Unify sites names
sites = Y_df["site"].to_numpy()

sites = unify_sites(sites, params.unify_sites)

if sites_use == "all":
    sites_use = np.unique(sites)

# TODO: Filter sites by size range
# TODO: subsample site to the smallests

X = X_df.to_numpy().astype(float)

# Set y
if problem_type == "binary_classification":
    female = Y_df["gender"]
    female.replace(to_replace={"F": 1, "M": 0}, inplace=True)
    female = np.array(female)
    y = female
else:
    age = np.round(Y_df["age"].to_numpy())
    y = age

# Check the target have at least 2 classes
if len(np.unique(y)) == 2:
    if problem_type != "binary_classification":
        raise ValueError(
            "The target has only 2 classes, please use binary_classification"
        )
elif problem_type != "regression":
    raise ValueError(
        "The target has more than 2 classes, please use regression"
    )

np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)

# Check por inf and NaN
assert not np.any(np.isnan(X))
assert not np.any(np.isinf(X))

# Keep originals
Xorig = np.copy(X)
yorig = np.copy(y)
sitesorig = np.copy(sites)

print(f"Using sites: {sites_use}")
# Select data form used sites
idx = np.zeros(len(sites))
for su in sites_use:
    idx = np.logical_or(idx, sites == su)
idx = np.where(idx)

X = X[idx]
y = y[idx]
sites = sites[idx]

# harmonization fails with low variance features
colvar = np.var(X, axis=0)
idxvar = np.argsort(-colvar)
idxvar = idxvar[range(0, n_high_var_feats)]
X = X[:, idxvar]

print("========= DATA INFO =========")
print(f" ORIG X SHAPE {Xorig.shape}")
print(f" ORIG Y SHAPE {yorig.shape}")
print(f" ORIG SITE SHAPE {sitesorig.shape}")
print(f" X SHAPE {X.shape}")
print(f" Y SHAPE {y.shape}")
print(f" SITE SHAPE {sites.shape}")
print("=============================")

usites, csites = np.unique(sites, return_counts=True)
print("Sites:")

# Check that at least 2 sites are used
assert len(usites) > 1
print(np.asarray((usites, csites)))
print(f"Data shape: {X.shape}")
print(f"{len(sites_use)} sites:")
sites_use, sites_count = np.unique(sites, return_counts=True)
assert len(sites_use) > 1
print(np.asarray((sites_use, sites_count)))

if problem_type == "binary_classification":
    print("Label counts:")
    uy, cy = np.unique(y, return_counts=True)
    print(np.asarray((uy, cy)))

show_hist(y, "y")
if len(sites_use) <= 3:
    for i in range(len(sites_use)):
        ii = sites == sites_use[i]
        show_hist(y[ii], f"y: {sites_use[i]} #{np.sum(ii)}")

# Out of Samples set up
Xoos = None
yoos = None
if sites_oos is not None:
    print(f"OOS sites: {sites_oos}")
    idx = np.zeros(len(sitesorig))
    for su in sites_oos:
        idx = np.logical_or(idx, sitesorig == su)
    idx = np.where(idx)
    Xoos = Xorig[idx]
    Xoos = Xoos[:, idxvar]
    yoos = yorig[idx]
    sitesoos = sitesorig[idx]
    covarsoos = None

if params.random_sites:
    print("\n*** SHUFFLING SITES ***\n")
    np.random.shuffle(sites)
    # induce site difference
    for ii, ss in enumerate(sites):
        if ss == usites[1]:
            X[ii] += np.random.normal(1.0, 1.0)
        else:
            X[ii] -= np.random.normal(-1.0, 1.0)


# ######################## Models Set ups
# #### Clasiffiers parameters
if problem_type == "binary_classification":
    if pred_model == "gssvm":
        params_svm = {
            "kernel": ("linear", "rbf"),
            "C": [0.01, 0.1, 1, 10, 100],
        }
        pred_model = GridSearchCV(SVC(probability=True), params_svm)
    elif pred_model == "rvc":
        pred_model = RVC(kernel="poly", degree=1)

    # if stack_model != "logit":
    #     raise ValueError(
    #         "Only logit is supported for binary classification stack model"
    #     )

else:
    # ['gsgpr', 'gssvm', 'ridgecv', 'rvr'],
    if pred_model == "gssvm":
        params_svm = {
            "kernel": ("linear", "rbf"),
            "C": [0.01, 0.1, 1, 10, 100],
        }
        pred_model = GridSearchCV(SVR(), params_svm)

    elif pred_model == "gsgpr":
        kernel1 = RBF() + WhiteKernel()
        kernel2 = DotProduct() + WhiteKernel()
        params_gpr = {"kernel": [kernel1, kernel2]}
        params_gpr = [
            {
                "alpha": [1e-5],
                "kernel": [RBF(l, (1e-7, 10e7)) for l in [0.1, 1, 10]],
            }
        ]
        pred_model = GridSearchCV(
            GPR(n_restarts_optimizer=5, normalize_y=True), params_gpr
        )
    elif pred_model == "ridgecv":
        pred_model = RidgeCV()
    elif pred_model == "rvr":
        pred_model = RVR(kernel="poly", degree=1)
    else:
        raise ValueError("Regression model not supported")

if pca:
    pca80 = PCA(n_components=0.8, svd_solver="full")
    pred_model = Pipeline(
        [("scaler", StandardScaler()), ("pca", pca80), ("model", pred_model)]
    )
elif scaler:
    pred_model = Pipeline([("scaler", RobustScaler()), ("model", pred_model)])

# Cross varidation Parameters
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


if harmonize_mode == "cheat":
    print("Cheat mode")
    cheat_model = JuHarmonize()
    X_harm = cheat_model.fit_transform(X, y, sites, covars)

    new_var = np.sum(X_harm.std(axis=0) < X.std(axis=0))
    # TODO: fix this or people will have strokes trying to understand
    print(
        f"Variance decreased for {new_var} features of {X.shape[1]}"
    )
    harmonize_mode = "none"

for i_fold, (train_index, test_index) in enumerate(kf.split(X)):
    if fold_to_do > 0 and i_fold != fold_to_do:
        continue
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
    )

    out_fold, acc_fold = eval_harmonizer(
        harm_model, X_test, y_test, sites_test, covars_test
    )

    out_fname = f"{harmonize_mode}_fold_{i_fold}_of_{n_splits}_out.csv"
    to_save = pd.DataFrame({"y_true": y_test, "y_pred": out_fold})
    to_save.to_csv(save_dir / out_fname, sep=';')