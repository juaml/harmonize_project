#%%
import numpy as np
import pandas as pd
#from lib.utils import  show_hist

def unify_sites(sites, unify_sites_names):

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

    return sites


def get_MRI_data(data_dir, unify_sites_names,problem_type, n_high_var_feats, sites_use, sites_oos, random_sites, covars):

    X_df = pd.read_csv(data_dir / "final_data" / "X_final.csv", header=0)
    X_df.reset_index(inplace=True)
    Y_df = pd.read_csv(data_dir / "final_data" / "Y_final.csv", header=0)
    Y_df.reset_index(inplace=True)

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

   #show_hist(y, "y")
    if len(sites_use) <= 3:
        for i in range(len(sites_use)):
            ii = sites == sites_use[i]
         #   show_hist(y[ii], f"y: {sites_use[i]} #{np.sum(ii)}")



    if random_sites:
        print("\n*** SHUFFLING SITES ***\n")
        np.random.shuffle(sites)
        # induce site difference
        for ii, ss in enumerate(sites):
            if ss == usites[1]:
                X[ii] += np.random.normal(1.0, 1.0)
            else:
                X[ii] -= np.random.normal(-1.0, 1.0)

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
        return X, y, sites, covars, Xoos, yoos, sitesoos, covarsoos
    else:
     return X, y, sites, covars