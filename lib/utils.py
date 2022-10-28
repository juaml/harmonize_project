
import numpy as np
import pandas as pd
import termplotlib as tpl

from .logging import logger


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
    logger.info(f'min size size: {min_size}')
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
            logger.info(f'excluding site: {su} #sub {n}')
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
    logger.info(f"Distribution of {main}")
    counts, bin_edges = np.histogram(y, bins=bins)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()


# def get_pred_acc(model, X, y, problem_type, sites=None, covars=None):
#     if problem_type == "binary_classification":
#         if sites is not None:
#             pred = model.predict_proba(X, sites, covars)[:, 1]
#         else:
#             pred = model.predict_proba(X)[:, 1]
#         acc = average_precision_score(y, pred)
#     else:
#         if sites is not None:
#             pred = model.predict(X, sites, covars)
#         else:
#             pred = model.predict(X)
#         acc = mean_absolute_error(y, pred)
#     return pred, acc

def check_params(params):
    valid_models = {
        "binary_classification": ["gssvm", "rvc", "svm"],
        "regression": ["gsgpr", "gssvm", "ridgecv", "rvr"],
    }

    if not params.pred_model in valid_models[params.problem_type]:
        ValueError("Invalid predict method")

    valid_harmonize_mode = ["none", "target","cheat", "notarget", "pretend", 
                            "pretend_nosite", "predict", "predict_pretend", 
                            "predict_pretend_nosite"]

    if not params.harmonize_mode in valid_harmonize_mode:
        ValueError("Invalid harmonization method")

    return 