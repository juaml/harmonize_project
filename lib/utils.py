from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    r2_score,
    mean_absolute_error,
)
from juharmonize import (
    JuHarmonize,
    JuHarmonizeRegressor,
    JuHarmonizeClassifier,
    JuHarmonizePredictor,
    JuHarmonizePredictorCV,
)
import numpy as np
import pandas as pd
import julearn
import termplotlib as tpl


def get_site_index(sites, sites_use):
    idx = np.zeros(len(sites))
    for su in sites_use:
        idx_su = pd.Series(sites).str.contains(fr'^{su}').to_numpy()
        assert np.sum(idx_su) > 0
        idx = np.logical_or(idx, idx_su)
    return idx


def sample_sites_smallest(sites, sites_use, sites_ignore=None):
    idx = np.zeros(len(sites))
    if sites_ignore is None:
        sites_ignore = np.array([])
    for su in sites_ignore:
        idx_su = sites == su
        idx = np.logical_or(idx, idx_su)
    min_size = np.inf
    for su in sites_use:
        idx_su = sites == su
        min_size = min(min_size, np.sum(idx_su))
    min_size = int(min_size)
    print(f'min size size: {min_size}')
    for su in sites_use:
        idx_su = sites == su
        if np.sum(idx_su) == min_size:
            idx = np.logical_or(idx, idx_su)
        if np.sum(idx_su) > min_size:
            idx_su = np.where(idx_su)[0]
            idx_su = np.random.choice(idx_su, min_size, replace=False)
            idx[idx_su] = 1
    return idx


def filter_site_size(sites, min_size, max_size=np.inf, sites_ignore=None):
    idx = np.zeros(len(sites))
    if sites_ignore is None:
        sites_ignore = np.array([])
    for su in sites_ignore:
        idx_su = sites == su
        idx = np.logical_or(idx, idx_su)
    sites_check = np.setdiff1d(np.unique(sites), sites_ignore)
    for su in sites_check:
        idx_su = sites == su
        n = np.sum(idx_su)
        if n >= min_size and n < max_size:
            idx = np.logical_or(idx, idx_su)
        else:
            print(f'excluding site: {su} #sub {n}')
    return idx


def get_groups(sites, y, bins=2):
    assert len(sites) == len(y)
    groups = np.array([""] * len(y), dtype=object)
    for su in np.unique(sites):
        idx = sites == su
        assert np.sum(idx) > 0
        suy = y[idx]
        subins = min(bins, len(np.unique(suy)))
        bin_edges = np.histogram_bin_edges(suy, bins=subins)
        bin_assgn = np.digitize(suy, bin_edges)
        binssu = [su + str(x) for x in bin_assgn]  # type: ignore
        groups[idx] = np.array(binssu)
    assert len(groups) == len(y)
    return groups


def show_hist(y, main='', bins=5):
    bins = min(bins, len(np.unique(y)))
    print(f"Distribution of {main}")
    counts, bin_edges = np.histogram(y, bins=bins)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()


def get_pred_acc(model, X, y, problem_type, sites=None, covars=None):
    if problem_type == "binary_classification":
        if sites is not None:
            pred = model.predict_proba(X, sites, covars)[:, 1]
        else:
            pred = model.predict_proba(X)[:, 1]
        acc = average_precision_score(y, pred)
    else:
        if sites is not None:
            pred = model.predict(X, sites, covars)
        else:
            pred = model.predict(X)
        acc = mean_absolute_error(y, pred)
    return pred, acc


def get_JuHarmonizeModel(problem_type, pred_model, stack_model,
                         n_splits=10, random_state=None, 
                         regression_points=50,
                         predict_ignore_site=False):
    if problem_type == "binary_classification":
        model = JuHarmonizeClassifier(
            pred_model=pred_model,
            n_splits=n_splits,
            stack_model=stack_model,
            use_cv_test_transforms=True,
            random_state=random_state,
            predict_ignore_site=predict_ignore_site)
    else:
        model = JuHarmonizeRegressor(
            pred_model=pred_model,
            n_splits=n_splits,
            regression_points=regression_points,
            stack_model=stack_model,
            use_cv_test_transforms=True,
            random_state=random_state,
            predict_ignore_site=predict_ignore_site)
    return model


def train_harmonizer(
        harm_type, X, y, sites, covars, pred_model,
        stack_model=None, random_state=None, n_splits=10):
    problem_type = "regression"
    if len(np.unique(y)) == 2:
        problem_type = "binary_classification"
    if isinstance(pred_model, str):
        _, pred_model = julearn.api.prepare_model(pred_model, problem_type)

    out = {}
    out['problem_type'] = problem_type
    out['harm_type']  = harm_type
    out['pred_model'] = pred_model
    out['harm_model'] = None
    if harm_type == 'none':
        out['pred_model'].fit(X, y)
    elif harm_type in ['target']:
        # harmonize train and then apply to test but using labels
        out['harm_model'] = JuHarmonize()
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        out['pred_model'].fit(X_harm, y)
    elif harm_type == 'notarget':
        out['harm_model'] = JuHarmonize(preserve_target=False)
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        out['pred_model'].fit(X_harm, y)
    elif harm_type == 'predict':
        out['harm_model'] = JuHarmonizePredictor()
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        out['pred_model'].fit(X_harm, y)
    elif harm_type in ['pretend', 'pretend_nosite']:
        assert stack_model is not None
        predict_ignore_site = harm_type == 'pretend_nosite'
        out['pred_model'] = get_JuHarmonizeModel(
            problem_type, pred_model, stack_model,
            predict_ignore_site=predict_ignore_site,
            random_state=random_state, n_splits=n_splits)
        out['pred_model'].fit(X, y, sites, covars)
    elif harm_type in ['predict_pretend', 'predict_pretend_nosite']:
        assert stack_model is not None
        predict_ignore_site = harm_type == 'pretend_nosite'
        out['pred_model'] = get_JuHarmonizeModel(
            problem_type, pred_model, stack_model,
            predict_ignore_site=predict_ignore_site,
            random_state=random_state, n_splits=n_splits)
        out['harm_model'] = JuHarmonizePredictorCV()
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        out['pred_model'].fit(X_harm, y, sites, covars)
    else:
        raise ValueError(f"Unknown harm_type {harm_type}")
    return out


def eval_harmonizer(harm, X, y, sites, covars=None):
    if harm['harm_type'] == 'none':
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] == 'target':
        assert y is not None
        X = harm['harm_model'].transform(X, y, sites, covars)
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] == 'notarget':
        X = harm['harm_model'].transform(X, y, sites, covars)
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] == 'predict':
        X = harm['harm_model'].transform(X)
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] in ['pretend', 'pretend_nosite']:
        out = harm['pred_model'].predict(X, sites, covars)
    elif harm['harm_type'] in ['predict_predict', 'predict_predict_nosie']:
        X = harm['harm_model'].transform(X)
        out = harm['pred_model'].predict(X, sites, covars)
    else:
        raise ValueError(f"Unknown harm_type: {harm['harm_type']}")

    if harm['problem_type'] == 'binary_classification':
        acc = accuracy_score(y, out)
    else:
        acc = r2_score(y, out)

    return out, acc
