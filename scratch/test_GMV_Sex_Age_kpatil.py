import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import accuracy_score, average_precision_score, r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
    ConstantKernel,
)
from sklearn.ensemble import RandomForestRegressor as RFR
from juharmonize import (
    JuHarmonize,
    JuHarmonizeRegressor,
    JuHarmonizeClassifier,
    JuHarmonizePredictor,
)
from juharmonize.utils import subset_data
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from skrvm import RVR, RVC
import copy
from util_kpatil import *

target = 'age'
min_site_size = 20
max_site_size = np.inf
n_high_var_feats = None
sample_to_smallest = False
scaler = True
pca = False
#sites_use = np.array(["1000Gehirne", "CamCAN", "eNKI"])
sites_use = np.array(["MPG_", "SWU_", "NKI_", "NYU_", "Utah_", "HNU_", \
                      "IPCAS_", "JHNU_",  "MRN_", "BNU_", "UPSM_", "LMU_", "BMB_", \
                      "IBATRT", "IACAS", "UWM", "UM", "XHCUMS"])
sites_use = np.array(["1000Gehirne", "eNKI"])
#sites_oos = None
sites_oos = np.array(["CamCAN"])
#sites_oos = np.array(["CamCAN", "eNKI", "IXI", "1000Gehirne"])
#sites_oos = np.array(["UM"])
datadir = '/data/project/harmonize/kpatil/data/s4_r8'
random_state1 = 3
random_state2 = 5

X = pd.read_csv(f'{datadir}/X_final.csv', header=0)
X.reset_index(inplace=True)
Y = pd.read_csv(f'{datadir}/Y_final.csv', header=0)
Y.reset_index(inplace=True)

# filter for adults
idx = Y['age'] >= 18
Y = Y[idx]
X = X[idx]
# filter for site size
idx = filter_site_size(Y['site'].to_numpy(), min_site_size, max_site_size, sites_oos)
Y = Y[idx]
X = X[idx]

if sample_to_smallest:
    sites = Y['site'].to_numpy()
    idx = sample_sites_smallest(sites, sites_use, sites_oos)
    Y = Y[idx]
    X = X[idx]

X = X.to_numpy()
n_obs_orig, n_feat_orig = X.shape
np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)
X = X[:, np.var(X, axis=0)>1e-5]
assert not np.any(np.isnan(X))
assert not np.any(np.isinf(X))

print(f'target is: {target}')
y = Y[target].to_numpy()
print(y.shape)
sites = Y['site'].to_numpy()
assert not np.any(np.isnan(y))
assert not np.any(np.isinf(y))


problem_type = "regression"
if len(np.unique(y)) == 2:
    problem_type = "binary_classification"
    uy = np.unique(y)
    y = np.array([0 if xx == uy[0] else 1 for xx in y])

print(f'this is a {problem_type} problem')

print(f"Using {len(sites_use)} sites: {sites_use}")
idx = np.zeros(len(sites))
for su in sites_use:
    idx_su = pd.Series(sites).str.contains(fr'^{su}').to_numpy()
    assert np.sum(idx_su)>0
    idx = np.logical_or(idx, idx_su)
idx = np.where(idx)

# get oos info before subsetting the data
Xoos = None
yoos = None
if sites_oos is not None:
    print(f"\nOOS sites: {sites_oos}")
    idxoos = np.zeros(len(sites))
    for su in sites_oos:
        idx_su = pd.Series(sites).str.contains(fr'^{su}').to_numpy()
        assert np.sum(idx_su)>0
        idxoos = np.logical_or(idxoos, idx_su)
    idxoos = np.where(idxoos)
    Xoos = X[idxoos]
    yoos = y[idxoos]
    sitesoos = sites[idxoos]
    covarsoos = None
    sites_oos = np.unique(sitesoos)

X = X[idx]
y = y[idx]
sites = sites[idx]
covars = None
sites_use = np.unique(sites)
# covars = pd.DataFrame({'age':age[idx]})

# harmonization fails with low variance features
if n_high_var_feats is None:
    n_high_var_feats = X.shape[1]

colvar = np.var(X, axis=0)
idxvar = np.argsort(-colvar)

if n_high_var_feats is None:
    idxvar = range(X.shape[1])
else:
    idxvar = idxvar[range(0, n_high_var_feats)]

X = X[:, idxvar]
if Xoos is not None:
     Xoos = Xoos[:, idxvar]

print(f"Data shape: {X.shape}")
print(f"{len(sites_use)} sites:")
sites_use, sites_count = np.unique(sites, return_counts=True)
assert len(sites_use) > 1
print(np.asarray((sites_use, sites_count)))

if problem_type == "binary_classification":
    print("Label counts:")
    uy, cy = np.unique(y, return_counts=True)
    print(np.asarray((uy, cy)))

show_hist(y, 'y')
if len(sites_use) <= 3:
    for i in range(len(sites_use)):
        ii = sites == sites_use[i]
        show_hist(y[ii], f'y: {sites_use[i]} #{np.sum(ii)}')

# print('\n*** SHUFFLING SITES ***\n')
# np.random.shuffle(sites)

model_full = JuHarmonize()
model_notarget = JuHarmonize(preserve_target=False)
model_harm = JuHarmonize()

params_svm = {"kernel": ("linear", "rbf"), "C": [0.01, 0.1, 1, 10, 100]}
kernel1 = RBF() + WhiteKernel()
kernel2 = DotProduct() + WhiteKernel()
params_gpr = {"kernel": [kernel1, kernel2]}
params_gpr = [{"alpha": [1e-5], "kernel": [RBF(l, (1e-7, 10e7)) for l in [0.1, 1, 10]]}]

if problem_type == "binary_classification":
    #model = GridSearchCV(SVC(probability=True), params_svm)
    model = RVC(kernel='poly', degree=1)
    stack_model_JuHaCV = "logit"
else:
    #model = GridSearchCV(GPR(n_restarts_optimizer=5, normalize_y=True), params_gpr)
    #model = GridSearchCV(SVR(), params_svm)
    #model = SVR()
    #model = RidgeCV()
    model = RVR(kernel='poly', degree=1)
    stack_model_JuHaCV = RFR() #RFR() #clone(model)  # 'svm' 'linreg' 'ridgecv' 'ridge' 'gpr'

if pca:
    pca80 = PCA(n_components = 0.8, svd_solver = 'full')
    model = Pipeline([("scaler", StandardScaler()), ("pca", pca80), ("model", model)])
elif scaler:
    model = Pipeline([('scaler', RobustScaler()), ('model', model)])

pred_model_cheat = clone(model)
pred_model_leak = clone(model)
pred_model_pool = clone(model)
pred_model_notarget = clone(model)
pred_model_JuHaCV = clone(model)
pred_model_pred = clone(model)

if problem_type == "binary_classification":
    eval = "average_precision_score"
    model_harm_cv = JuHarmonizeClassifier(
        pred_model=pred_model_JuHaCV,
        n_folds=10,
        stack_model=stack_model_JuHaCV,
        use_cv_test_transforms=True,
        random_state=random_state1)
else:
    eval = "mae"
    model_harm_cv = JuHarmonizeRegressor(
        pred_model=pred_model_JuHaCV,
        n_folds=10,
        regression_points=50,
        stack_model=stack_model_JuHaCV,
        use_cv_test_transforms=True,
        random_state=random_state1)

model_harm_cv_pred = copy.deepcopy(model_harm_cv)

X_harm = model_full.fit_transform(X, y, sites, covars)
print(
    f"Variance decreased for {np.sum(np.std(X_harm, axis=0) < np.std(X, axis=0))} features of {X.shape[1]}"
)
harm_pred = JuHarmonizePredictor()

if Xoos is not None:
    print(f"OOS sites: {sites_oos}, #sub: {Xoos.shape[0]}")
    show_hist(yoos, 'yoos')

    pred_model_pool.fit(X, y)
    pred_pool = pred_model_pool.predict(Xoos)

    harm_pred = JuHarmonizePredictor()
    harm_pred.fit(X, X_harm)

    Xoospred = harm_pred.predict(Xoos)
    pred_model_pred.fit(X_harm, y)
    pred_pred = pred_model_pred.predict(Xoospred)

    # set up model_harm_cv for OOS
    model_harm_cv.predict_ignore_site = True
    model_harm_cv_pred.predict_ignore_site = True
    #if problem_type == "regression":
    #    model_harm_cv.regression_points = 10
    #    #model_harm_cv.stack_model = RFR()
    model_harm_cv.fit(X, y, sites, covars)

    XX = JuHarmonizePredictorCV(X, y, sites, covars)
    model_harm_cv_pred.fit(XX, y, sites, covars)
    if problem_type == "binary_classification":
        pred_noleak = model_harm_cv.predict_proba(Xoos, sitesoos, covarsoos)[:,1]
        pred_noleak_pred = model_harm_cv_pred.predict_proba(Xoospred, sitesoos, covarsoos)[:,1]
    else:
        pred_noleak = model_harm_cv.predict(Xoos, sitesoos, covarsoos)
        pred_noleak_pred = model_harm_cv.predict(Xoospred, sitesoos, covarsoos)

    for su in np.unique(sites_oos):
        idx = pd.Series(sitesoos).str.contains(fr'^{su}').to_numpy()
        print(f"OOS site: {su}, #sub {np.sum(idx)}")
        if problem_type == "binary_classification":
            acc_noleak = average_precision_score(yoos[idx], pred_noleak[idx])
            acc_pool = average_precision_score(yoos[idx], pred_pool[idx])
            acc_pred = average_precision_score(yoos[idx], pred_pred[idx])
            acc_noleak_pred = average_precision_score(yoos[idx], pred_noleak_pred[idx])
        else:
            acc_noleak = mae(yoos[idx], pred_noleak[idx])
            acc_pool = mae(yoos[idx], pred_pool[idx])
            acc_pred = mae(yoos[idx], pred_pred[idx])
            acc_noleak_pred = mae(yoos[idx], pred_noleak_pred[idx])

        print(f"OOS performance: pool={acc_pool}")
        print(f"OOS performance: pred={acc_pred}")
        print(f"OOS performance: noleak={acc_noleak}")
        print(f"OOS performance: noleak_pred={acc_noleak_pred}")


model_harm_cv.predict_ignore_site = False
model_harm_cv.regression_points = 50

model_harm_cv_pred.predict_ignore_site = False
model_harm_cv_pred.regression_points = 50

# test Harmonization in CV
n_splits = 3
#groups = get_groups(sites, y)
#groups = sites
#group_kfold = GroupKFold(n_splits=n_splits)
#kf = group_kfold.split(X, y, groups)
kf = KFold(n_splits=n_splits, random_state=random_state2, shuffle=True).split(X)
for i_fold, (train_index, test_index) in enumerate(kf):
    print(f"Fold {i_fold+1}/{n_splits}: #train {len(train_index)}, #test {len(test_index)}")
    X_train, sites_train, y_train, covars_train = subset_data(
        train_index, X, sites, y, covars)

    X_test, sites_test, y_test, covars_test = subset_data(
        test_index, X, sites, y, covars)

    usites, csites = np.unique(sites_train, return_counts=True)
    print(usites)
    print(csites)

    # Harmonize without target
    X_train_harm_notarget = model_notarget.fit_transform(
        X_train, y_train, sites_train, covars_train)
    X_test_harm_notarget = model_notarget.transform(
        X_test, y_test, sites_test, covars_test)

    # Harmonize X using leak
    X_train_harm = model_harm.fit_transform(
        X_train, y_train, sites_train, covars_train)
    X_test_harm = model_harm.transform(X_test, y_test, sites_test, covars_test)
    assert not np.any(np.isnan(X_train_harm))
    assert not np.any(np.isinf(X_train_harm))

    # Harmonize using prediction
    harm_pred.fit(X_train, X_train_harm)
    X_test_harm_pred = harm_pred.predict(X_test)

    # Harmonize X with no leak
    model_harm_cv.fit(X_train, y_train, sites_train, covars_train)
    #X_test_harm_cv = model_harm_cv.transform(X_test, sites_test, covars_test)
    XX = JuHarmonizePredictorCV(X_train, y_train, sites_train, covars_train)
    model_harm_cv_pred.fit(XX, y_train, sites_train, covars_train)

    # Fit model
    pred_model_cheat.fit(X_harm[train_index], y_train)
    pred_model_leak.fit(X_train_harm, y_train)
    pred_model_pred.fit(X_train_harm, y_train)
    pred_model_pool.fit(X_train, y_train)
    pred_model_notarget.fit(X_train_harm_notarget, y_train)

    # harmonized without target
    pred_notarget, acc_notarget = get_pred_acc(pred_model_notarget, X_test_harm_notarget,
            y_test, problem_type)
    # leaky way 1: complete data harmonization
    pred_cheat, acc_cheat = get_pred_acc(pred_model_cheat, X_harm[test_index],
            y_test, problem_type)
    # leaky way 1: labels in test set
    pred_leak, acc_leak = get_pred_acc(pred_model_leak, X_test_harm,
            y_test, problem_type)
    # pool
    pred_pool, acc_pool = get_pred_acc(pred_model_pool, X_test,
            y_test, problem_type)
    # noleak
    pred_noleak, acc_noleak = get_pred_acc(model_harm_cv, X_test,
            y_test, problem_type, sites_test, covars_test)
    # pred
    pred_pred, acc_pred = get_pred_acc(pred_model_pred, X_test_harm_pred,
            y_test, problem_type)
    # noleak pred
    pred_noleak_pred, acc_noleak_pred = get_pred_acc(model_harm_cv_pred,
            X_test_harm_pred, y_test, problem_type, sites_test, covars_test)


    print(f"notarget: {acc_notarget:.4f}")
    print(f"cheat   : {acc_cheat:.4f}")
    print(f"leak    : {acc_leak:.4f}")
    print(f"pool    : {acc_pool:.4f}")
    print(f"pred    : {acc_pred:.4f}")
    print(f"noleak  : {acc_noleak:.4f}")
    print(f"noleakpred: {acc_noleak_pred:.4f}")
    print("-----------------------")

