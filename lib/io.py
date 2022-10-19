

import numpy as np
import pandas as pd
from pathlib import Path

from .utils import show_hist
from .logging import logger


def unify_sites(sites, unify_sites_names):
    logger.info(f"Unifying sites {unify_sites_names}")
    # make all names uppercase
    unify_sites_names = [element.upper() for element in unify_sites_names]

    # Transform to DF to use isin function
    unify_sites_names = pd.DataFrame(unify_sites_names)

    # Unify the IXI sites
    if unify_sites_names.isin(["IXI"]).any()[0]:

        sites.replace(
            to_replace={"IXI/Guys": "IXI", "IXI/HH": "IXI", "IXI/IOP": "IXI"},
            inplace=True,
        )

    # Unify the AOMIC sites
    if unify_sites_names.isin(["AOMIC"]).any()[0]:

        sites.replace(
            to_replace={"ID1000": "AOMIC", "PIOP1": "AOMIC", "PIOP2": "AOMIC"},
            inplace=True,
        )

    # Unify the CORR sites
    if unify_sites_names.isin(["CORR"]).any()[0]:
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

    logger.info("Sites unified")
    return sites


def set_argparse_params(parser, use_oos=False):
    parser.add_argument("--data_dir", type=str, default="",
                        help="Data store path")
    parser.add_argument(
        "--n_high_var_feats", type=int, default=100,
        help="High variance features"
    )
    parser.add_argument(
        "--unify_sites",
        nargs="+",
        type=str,
        default=["IXI", "CORR", "AOMIC"],
        help="Sites to unify",
    )
    parser.add_argument(
        "--sites_use", nargs="+", type=str, default=["all"], help="Used sites"
    )

    parser.add_argument(
        "--sites_oos",
        nargs="+",
        type=str,
        required=True,
        help="Out of sample sites",
    )
    parser.add_argument(
        "--covars", type=str, default=None, help="If randomized sites"
    )
    parser.add_argument(
        "--random_sites", type=bool, default=False, help="If randomized sites"
    )
    return parser


def get_MRI_data(params, problem_type, use_oos=False):

    data_dir = Path(params.data_dir)
    logger.info(f"Loading data from {data_dir.as_posix()}")
    unify_sites_names = params.unify_sites
    n_high_var_feats = params.n_high_var_feats
    sites_use = params.sites_use

    sites_oos = None
    if use_oos is True:
        sites_oos = params.sites_oos
        logger.info(f"Sites OOS: {sites_oos}")
        if sites_oos is None:
            raise ValueError("Out of sample sites not specified")
    random_sites = params.random_sites
    covars = params.covars

    logger.info("Reading X...")
    X_df = pd.read_csv(data_dir / "final_data" / "X_final.csv", header=0)
    X_df.reset_index(inplace=True)
    logger.info("Reading X done")
    logger.info("Reading Y...")
    Y_df = pd.read_csv(data_dir / "final_data" / "Y_final.csv", header=0)
    Y_df.reset_index(inplace=True
    logger.info("Reading Y done"))

    # ############## Format data
    # Unify sites names
    sites = Y_df["site"]

    sites = unify_sites(sites, unify_sites_names)

    # put variables in the right format
    sites = np.array(sites)

    if sites_use == "all":
        sites_use = np.unique(sites)

    # TODO: Filter sites by size range
    # TODO: subsample site to the smallests

    X = X_df.to_numpy().astype(float)

    # Set y
    if problem_type == "binary_classification":
        logger.info("Converting 'gender' to binary classification")
        female = Y_df["gender"]
        female.replace(to_replace={"F": 1, "M": 0}, inplace=True)
        female = np.array(female)
        y = female
    else:
        logger.info("Rounding up age")
        age = np.round(Y_df["age"].to_numpy())
        y = age

    # Check the target have at least 2 classes
    if len(np.unique(y)) == 2:
        if problem_type != "binary_classification":
            raise ValueError(
                "The target has only 2 classes, please use "
                "binary_classification"
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

    logger.info(f"Using sites: {sites_use}")
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

    logger.info("========= DATA INFO =========")
    logger.info(f" ORIG X SHAPE {Xorig.shape}")
    logger.info(f" ORIG Y SHAPE {yorig.shape}")
    logger.info(f" ORIG SITE SHAPE {sitesorig.shape}")
    logger.info(f" X SHAPE {X.shape}")
    logger.info(f" Y SHAPE {y.shape}")
    logger.info(f" SITE SHAPE {sites.shape}")
    logger.info("=============================")

    usites, csites = np.unique(sites, return_counts=True)
    logger.info("Sites:")

    # Check that at least 2 sites are used
    assert len(usites) > 1
    logger.info(np.asarray((usites, csites)))
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"{len(sites_use)} sites:")
    sites_use, sites_count = np.unique(sites, return_counts=True)
    assert len(sites_use) > 1
    logger.info(np.asarray((sites_use, sites_count)))

    if problem_type == "binary_classification":
        logger.info("Label counts:")
        uy, cy = np.unique(y, return_counts=True)
        logger.info(np.asarray((uy, cy)))

    show_hist(y, "y")
    if len(sites_use) <= 3:
        for i in range(len(sites_use)):
            ii = sites == sites_use[i]
            show_hist(y[ii], f"y: {sites_use[i]} #{np.sum(ii)}")

    if random_sites:
        logger.info("\n*** SHUFFLING SITES ***\n")
        np.random.shuffle(sites)
        # induce site difference
        for ii, ss in enumerate(sites):
            if ss == usites[1]:
                X[ii] += np.random.normal(1.0, 1.0)
            else:
                X[ii] -= np.random.normal(-1.0, 1.0)

    if sites_oos is not None:
        logger.info(f"Setting OOS sites {sites_oos}")
        # Out of Samples set up
        Xoos = None
        yoos = None
        logger.info(f"OOS sites: {sites_oos}")
        idx = np.zeros(len(sitesorig))
        for su in sites_oos:
            idx = np.logical_or(idx, sitesorig == su)
        idx = np.where(idx)
        Xoos = Xorig[idx]
        Xoos = Xoos[:, idxvar]
        yoos = yorig[idx]
        sitesoos = sitesorig[idx]
        covarsoos = None
        out = X, y, sites, covars, Xoos, yoos, sitesoos, covarsoos
    else:
        out = X, y, sites, covars
    logger.info("Data reading done!")
    return out
