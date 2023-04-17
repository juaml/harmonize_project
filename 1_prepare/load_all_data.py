# %%

import numpy as np
import pandas as pd


def dataset_difference(main_dir, sites_use, data_name, data):

    data = data[data['site'].isin(sites_use)]

    subj = data["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    # X = X.loc[X[1].isin(subj)]

    print("For Site: " + sites_use[0])
    print("Data: " + str(X.shape[0]))
    print("MetaData: " + str(subj.shape[0]))
    print("Data Different: " + str(X.shape[0]-subj.shape[0]))

    return


def get_dataset(main_dir, sites_use, data_name, metadata):

    metadata = metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X, metadata


def get_1000_dataset(main_dir, sites_use, data_name, metadata):

    metadata = metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn, sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn]).split("_", 1)[0]

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X, metadata


def get_CoRR_dataset(main_dir, sites_use, data_name, metadata):

    metadata = metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn, sub in enumerate(X[1]):

        id = str(X[1][nn]).split("_", 1)[0]

        X[1][nn] = id.split("-", 1)[1]

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X, metadata


def get_AOMIC_PIOP2(main_dir, sites_use, data_name, metadata):

    metadata = metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn, sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn])+"_site-PIOP2"

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X, metadata


def get_AOMIC_1000(main_dir, sites_use, data_name, metadata):

    metadata = metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn, sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn])+"_site-ID1000"

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X, metadata


def get_AOMIC_PIOP1(main_dir, sites_use, data_name, metadata):

    metadata = metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn, sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn])+"_site-PIOP1"

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X, metadata


def get_OASIS3_dataset(main_dir, sites_use, data_name, metadata):

    metadata = metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn, sub in enumerate(X[1]):

        id = str(X[1][nn]).split("_", 1)[0]

        X[1][nn] = id.split("-", 1)[1]

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X, metadata


def save_dataset(X, Y, sites_use, save_dir):
    # Sort everything with subjects
    X_final = X.sort_values(1)
    Y_final = Y.sort_values("subject")
    subject_Y = Y_final["subject"].reset_index()
    subject_X = X_final[1].reset_index()

    assert subject_Y["subject"].equals(subject_X[1])
    # delete sites, subject for the X data
    X_final = X_final.drop([0, 1], axis=1)

    Y_final = Y_final[["site", "subject", "age", "gender"]]
    print(sites_use+" :" + str(len(np.unique(Y_final["subject"]))))
    X_final.to_csv(save_dir+"X_"+data_name+".csv", index=False)
    Y_final.to_csv(save_dir+"Y_"+data_name+".csv", index=False)

    return


############################################
metadata_dir = "/home/nnieto/Nico/Harmonization/data/"
metadata_name = "meta_data"
metadata = pd.read_csv(metadata_dir+metadata_name)
main_dir = "/data/project/harmonize/data/CAT/s4_r8"
save_dir = "/data/project/harmonize/data/CAT/s4_r8/final_data_split"
############################################

sites_use = ["1000Gehirne"]
data_name = "1000brains.txt"
X_1000Gehirne, Y_1000Gehirne = get_1000_dataset(main_dir, sites_use, data_name, metadata)
save_dataset(X_1000Gehirne, Y_1000Gehirne, sites_use, save_dir)
############################################

sites_use = ["CamCAN"]
data_name = "CamCAN.txt"
X_CamCAN, Y_CamCAN = get_dataset(main_dir, sites_use, data_name, metadata)
save_dataset(X_CamCAN, Y_CamCAN, sites_use, save_dir)
############################################

sites_use = ['BMB_1', 'BNU_1', 'BNU_2', 'BNU_3', 'HNU_1',
             'IACAS', 'IBATRT', 'IPCAS_1', 'IPCAS_2', 'IPCAS_3', 'IPCAS_4',
             'IPCAS_5', 'IPCAS_6', 'IPCAS_7', 'IPCAS_8',  'JHNU_1', 'LMU_1',
             'LMU_2', 'LMU_3', 'MPG_1', 'MRN_1', 'NKI_1', 'NYU_1', 'NYU_2',
             'SWU_1', 'SWU_2', 'SWU_3', 'SWU_4', 'UM', 'UPSM_1',
             'UWM', 'Utah_1', 'Utah_2', 'XHCUMS']

data_name = "CoRR.txt"
X_Corr, Y_Corr = get_CoRR_dataset(main_dir, sites_use, data_name, metadata)
sites_use = ["CoRR"]
save_dataset(X_Corr, Y_Corr, sites_use, save_dir)
############################################

sites_use = ["HCP"]
data_name = "hcp.txt"
X_HCP, Y_HCP = get_dataset(main_dir, sites_use, data_name, metadata)
save_dataset(X_HCP, Y_HCP, sites_use, save_dir)
############################################

sites_use = ["IXI/Guys", "IXI/HH", "IXI/IOP"]
data_name = "ixi.txt"
X_IXI, Y_IXI = get_dataset(main_dir, sites_use, data_name, metadata)
sites_use = ["IXI"]
save_dataset(X_IXI, Y_IXI, sites_use, save_dir)
############################################

sites_use = ["OASIS3"]
data_name = "OASIS3.txt"
X_OASIS3, Y_OASIS3 = get_OASIS3_dataset(main_dir, sites_use, data_name, metadata)
save_dataset(X_OASIS3, Y_OASIS3, sites_use, save_dir)

############################################

sites_use = ["ID1000"]
data_name = "aomic.txt"
X_AOMIC, Y_AOMIC = get_AOMIC_1000(main_dir, sites_use, data_name, metadata)
save_dataset(X_AOMIC, Y_AOMIC, sites_use, save_dir)
############################################

sites_use = ["PIOP1"]
data_name = "aomic-piop1.txt"
X_AOMIC_PIOP1, Y_AOMIC_PIOP1 = get_AOMIC_PIOP1(main_dir, sites_use, data_name, metadata)
save_dataset(X_AOMIC_PIOP1, Y_AOMIC_PIOP1, sites_use, save_dir)
############################################

sites_use = ["PIOP2"]
data_name = "aomic-piop2.txt"
X_AOMIC_PIOP2, Y_AOMIC_PIOP2 = get_AOMIC_PIOP2(main_dir, sites_use, data_name, metadata)
save_dataset(X_AOMIC_PIOP2, Y_AOMIC_PIOP2, sites_use, save_dir)
############################################

sites_use = ["eNKI"]
data_name = "eNKI.txt"
X_eNKI, Y_eNKI = get_1000_dataset(main_dir, sites_use, data_name, metadata)
save_dataset(X_eNKI, Y_eNKI, sites_use, save_dir)
# %%
import pandas as pd
main_dir = "/home/nnieto/Nico/Harmonization/data/Extracted_s4r8_data/"
save_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
metadata_dir = "/home/nnieto/Nico/Harmonization/datasets_metadata_local/datasets_metadata/"


def get_SALD_dataset(main_dir, metadata_dir):
    data_name = "SALD.txt"
    metadata_name = "SALD_information.csv"

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)
    metadata = pd.read_csv(metadata_dir+metadata_name, sep=" ")

    for nn, sub in enumerate(X[1]):

        id = str(X[1][nn]).split("_", 1)[0]

        X[1][nn] = id.split("-", 1)[1]

    sub_x = X[1].astype(int)
    sub_y = metadata["Sub_ID"]

    X = X.iloc[:, 2:]
    if not any(sub_x == sub_y):
        Warning("Wrong aligment in dataframes")

    # Put metadata in order
    metadata["site"] = "SALD"
    metadata.rename(columns={"Sex": "gender"}, inplace=True)
    metadata["gender"].replace({"M ": "M"}, inplace=True)
    metadata.rename(columns={"Age": "age"}, inplace=True)
    metadata.rename(columns={"Sub_ID": "subject"}, inplace=True)
    y = metadata.loc[:, ["site", "subject", "gender", "age"]]

    return X, y

data_name = "SALD"
X_SALD, y_SALD = get_SALD_dataset(main_dir, metadata_dir)
X_SALD.to_csv(save_dir+"X_"+data_name+".csv", index=False)
y_SALD.to_csv(save_dir+"Y_"+data_name+".csv", index=False)
# %%

def get_SALD_dataset(main_dir, metadata_dir):

    data_name = "DLBS.txt"
    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)
    for nn, sub in enumerate(X[1]):

        id = str(X[1][nn]).split("_ses", 1)[0]

        X[1][nn] = id.split("-00", 1)[1]

    metadata_name = "DALLAS_age.csv"
    meta_age = pd.read_csv(metadata_dir + metadata_name)

    metadata_name = "DALLAS_gender.csv"
    meta_gender = pd.read_csv(metadata_dir + metadata_name)

    metadata = pd.merge(meta_gender, meta_age, how="inner", on="Subject")
    metadata["site"] = "DLBS"
    metadata.rename(columns={"M/F": "gender"}, inplace=True)
    metadata.rename(columns={"Age": "age"}, inplace=True)
    metadata.rename(columns={"Subject": "subject"}, inplace=True)

    y = metadata.loc[:, ["site", "gender", "age", "subject"]]

    sub_x = X[1].astype(int)
    sub_y = y["subject"]

    X = X.iloc[:, 2:]
    if not any(sub_x == sub_y):
        Warning("Wrong aligment in dataframes")

    return X, y
# %%


X_DLBS, y_DLBS = get_SALD_dataset(main_dir, metadata_dir)
data_name = "DLBS"
X_DLBS.to_csv(save_dir+"X_"+data_name+".csv", index=False)
y_DLBS.to_csv(save_dir+"Y_"+data_name+".csv", index=False)
# %%
