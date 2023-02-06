
import numpy as np
import pandas as pd
import termplotlib as tpl
import matplotlib.pyplot as plt
import seaborn as sbn
from .logging import logger
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    f1_score,
    roc_auc_score,
)


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

    valid_harmonize_mode = ["none", "target", "cheat", "notarget", "pretend",
                            "pretend_nosite", "predict", "predict_pretend",
                            "predict_pretend_nosite"]

    if not params.harmonize_mode in valid_harmonize_mode:
        ValueError("Invalid harmonization method")

    return


def remove_extreme_TIV(X, Y, TIV_percentage):

    # Get each gender
    male = Y[Y["gender"] == "M"]
    female = Y[Y["gender"] == "F"]

    # Select males
    num_to_delete = male.shape[0] - np.round(male.shape[0] * TIV_percentage / 100).astype(int)

    gender_TIV = male["TIV"]

    sort_index = np.argsort(gender_TIV.to_numpy())

    TIV_to_remove = male.iloc[sort_index[num_to_delete], 4]

    mask = np.where(gender_TIV > TIV_to_remove)

    male.reset_index(inplace=True)

    males_to_keep = male.drop(mask[0])

    # select females
    num_to_delete = np.round(female.shape[0] * TIV_percentage / 100).astype(int)

    gender_TIV = female["TIV"]

    sort_index = np.argsort(gender_TIV.to_numpy())

    TIV_to_remove = female.iloc[sort_index[num_to_delete], 4]

    mask = np.where(gender_TIV < TIV_to_remove)

    female.reset_index(inplace=True)
    females_to_keep = female.drop(mask[0])

    index_to_keep = pd.concat([males_to_keep["index"], females_to_keep["index"]])

    y_tvi = Y.drop(index_to_keep)

    Y_final = Y.drop(y_tvi.index)
    X_final = X.drop(y_tvi.index)

    return X_final, Y_final


def table_generation(data, stats=["Age_bias", "R2", "MAE"]):

    harm_modes = np.unique(data["harmonize_mode"])
    table = pd.DataFrame(columns=harm_modes, index=stats)

    for mode in harm_modes:
        resut_mode = data[data["harmonize_mode"] == mode]

        predicted_age = resut_mode["y_pred"]
        true_age = resut_mode["y_true"]

        final_stat = []
        for stat in stats:
            if stat == "Age_bias":
                age_bias = np.corrcoef(true_age, predicted_age-true_age)[0, 1]
                final_stat = np.append(final_stat, age_bias)
            elif stat == "R2":
                r2_data = r2_score(true_age, predicted_age)
                final_stat = np.append(final_stat, r2_data)
            elif stat == "MAE":
                error_data = mean_absolute_error(true_age, np.round(predicted_age))
                final_stat = np.append(final_stat, error_data)
            elif stat == "ACC":
                error_data = accuracy_score(true_age, np.round(predicted_age))
                final_stat = np.append(final_stat, error_data)

        table[mode] = final_stat

    return table.T


def plot_barplot(data, exlude_notarget=True, absolute_error=True):
    harm_modes = np.unique(data["harmonize_mode"]).tolist()
    if exlude_notarget:
        harm_modes.remove("notarget")

    final_stat = []
    for mode in harm_modes:
        resut_mode = data[data["harmonize_mode"] == mode]
        predicted_age = resut_mode["y_pred"]
        true_age = resut_mode["y_true"]
        error_data = mean_absolute_error(true_age, np.round(predicted_age))
        final_stat = np.append(final_stat, error_data)

    to_sort = [harm_modes, final_stat]
    df = pd.DataFrame(to_sort, index=["method", "value"])
    df = df.sort_values(by="value", axis=1)
    df = df.T
    sort_mode = df["method"]

    if absolute_error:
        data["y_diff"] = np.abs(data["y_true"] - np.round(data["y_pred"]))
    else:
        data["y_diff"] = data["y_true"] - np.round(data["y_pred"])

    plt.figure(figsize=[30, 15])
    sbn.boxenplot(data, y="y_diff", x="harmonize_mode", order=sort_mode)

    return


def plot_barplot_classification(data, exlude_notarget=True):
    harm_modes = np.unique(data["harmonize_mode"]).tolist()
    if exlude_notarget:
        harm_modes.remove("notarget")

    data["acc"] = data["y_pred"]*0
    final_stat = []
    for mode in harm_modes:
        resut_mode = data[data["harmonize_mode"] == mode]
        predicted_age = resut_mode["y_pred"]
        true_age = resut_mode["y_true"]
        error_data = accuracy_score(true_age, np.round(predicted_age))
        data.loc[data["harmonize_mode"] == mode, "acc"] = error_data
        final_stat = np.append(final_stat, error_data)

    to_sort = [harm_modes, final_stat]
    df = pd.DataFrame(to_sort, index=["method", "value"])
    df = df.sort_values(by="value", axis=1)
    df = df.T
    sort_mode = df["method"]

    predicted_age = resut_mode["y_pred"]
    true_age = resut_mode["y_true"]

    plt.figure(figsize=[30, 15])
    sbn.barplot(data, y="acc", x="harmonize_mode", order=sort_mode)

    return


def plot_grup_barplot(data, exlude_notarget=True, absolute_error=True,
                      harm_modes=None):

    if harm_modes is None:
        harm_modes = np.unique(data["harmonize_mode"]).tolist()

    if exlude_notarget:
        # Check the experiment have no target
        if "notarget" in harm_modes:
            harm_modes.remove("notarget")

    final_stat = []
    for mode in harm_modes:
        resut_mode = data[data["harmonize_mode"] == mode]
        predicted_age = resut_mode["y_pred"]
        true_age = resut_mode["y_true"]
        error_data = mean_absolute_error(true_age, np.round(predicted_age))
        final_stat = np.append(final_stat, error_data)

    to_sort = [harm_modes, final_stat]
    df = pd.DataFrame(to_sort, index=["method", "value"])
    df = df.sort_values(by="value", axis=1)
    df = df.T
    sort_mode = df["method"]

    if absolute_error:
        data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
    else:
        data["y_diff"] = data["y_true"]-data["y_pred"]

    g = sbn.catplot(
        data=data, kind="boxen",
        x="site", y="y_diff", hue="harmonize_mode",
        height=6, hue_order=sort_mode, legend=harm_modes
    )
    g.set_axis_labels("", "Prediction difference")
    plt.xlabel("Dataset")
    plt.title("Harmonization schemes comparison")
    plt.grid(alpha=0.5, axis="y", c="black")

    return


def extract_experiment_data(exp_dir, exp_name, train_acc=False):
    for experiment in exp_name:
        in_path = Path(exp_dir) / experiment
        all_dfs = []
        if train_acc:
            files = '*train.csv'

        else:
            files = '*out.csv'
        for t_fname in in_path.glob(files):
            all_dfs.append(pd.read_csv(t_fname, sep=';'))

        results_df = pd.concat(all_dfs)
    return results_df


def extract_experiment_data_oos(exp_dir, exp_name, train_acc=False):
    for experiment in exp_name:
        in_path = Path(exp_dir) / experiment
        all_dfs = []
        if train_acc:
            files = '*train.csv'

        else:
            files = '*out.csv'

        for t_fname in in_path.glob(files):
            df = pd.read_csv(t_fname, sep=';')

            all_dfs.append(df)

        results_df = pd.concat(all_dfs)

    return results_df


def classification_table(data, harm_modes=None, stats=["acc"]):

    if harm_modes is None:
        harm_modes = np.unique(data["harmonize_mode"])

    table = pd.DataFrame(columns=harm_modes, index=stats)

    for mode in harm_modes:
        resut_mode = data[data["harmonize_mode"] == mode]

        predicted_y = resut_mode["y_pred"]
        true_y = resut_mode["y_true"]

        final_stat = []
        for stat in stats:
            if stat == "acc":
                acc = accuracy_score(true_y, np.round(predicted_y))
                final_stat = np.append(final_stat, acc)

            if stat == "F1_score":
                F1 = f1_score(true_y, predicted_y)
                final_stat = np.append(final_stat, F1)

            if stat == "auc":
                auc = roc_auc_score(true_y, predicted_y)
                final_stat = np.append(final_stat, auc)

        table[mode] = final_stat

    return table.T


def get_fold_acc_auc(results_df):
    # Compute ACC and AUC for each fold.
    acc_fold = []
    data_final = pd.DataFrame(columns=["acc", "auc", "site",
                                       "Harmonization Schemes"])
    site_fold = []
    harm_fold = []
    auc_fold = []
    # For each site
    for site in np.unique(results_df["site"]):
        results_site = results_df[results_df["site"] == site]
        # For each fold
        for fold in np.unique(results_site["fold"]):
            results_fold = results_site[results_site["fold"] == fold]
            # For each mehods
            for harm in np.unique(results_fold["harmonize_mode"]):

                results_harm = results_fold[results_fold["harmonize_mode"] == harm]
                # Compute ACC
                acc_fold.append(accuracy_score(results_harm["y_true"],
                                np.round(results_harm["y_pred"])))
                # Compute AUC
                auc_fold.append(roc_auc_score(results_harm["y_true"],
                                results_harm["y_pred"]))
                site_fold.append(site)
                harm_fold.append(harm)

    # Create final Dataframe
    data_final["acc"] = np.array(acc_fold)
    data_final["auc"] = np.array(acc_fold)
    data_final["site"] = site_fold
    data_final["Harmonization Schemes"] = harm_fold
    return data_final
