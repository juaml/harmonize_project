#%%

import numpy as np
import pandas as pd

def dataset_difference(main_dir,sites_use,data_name,data):

    data= data[data['site'].isin(sites_use)]

    subj = data["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    # X = X.loc[X[1].isin(subj)]

    print("For Site: "+ sites_use[0])
    print("Data: " + str(X.shape[0]))
    print("MetaData: " + str(subj.shape[0]))
    print("Data Different: " + str(X.shape[0]-subj.shape[0]))

    return


def get_dataset(main_dir,sites_use,data_name,metadata):

    metadata= metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X , metadata


def get_1000_dataset(main_dir,sites_use,data_name,metadata):

    metadata= metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn,sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn]).split("_",1)[0]

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X , metadata


def get_CoRR_dataset(main_dir,sites_use,data_name,metadata):

    metadata= metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn,sub in enumerate(X[1]):

        id = str(X[1][nn]).split("_",1)[0]

        X[1][nn] = id.split("-",1)[1]

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X , metadata


def get_AOMIC_PIOP2(main_dir,sites_use,data_name,metadata):

    metadata= metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn,sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn])+"_site-PIOP2"

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X , metadata


def get_AOMIC_1000(main_dir,sites_use,data_name,metadata):

    metadata= metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn,sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn])+"_site-ID1000"

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X , metadata


def get_AOMIC_PIOP1(main_dir,sites_use,data_name,metadata):

    metadata= metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn,sub in enumerate(X[1]):
        X[1][nn] = str(X[1][nn])+"_site-PIOP1"

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X , metadata

def get_OASIS3_dataset(main_dir,sites_use,data_name,metadata):

    metadata= metadata[metadata['site'].isin(sites_use)]

    subj = metadata["subject"]

    X = pd.read_csv(main_dir+data_name, sep=" ", header=None)

    for nn,sub in enumerate(X[1]):

        id = str(X[1][nn]).split("_",1)[0]

        X[1][nn] = id.split("-",1)[1]

    X = X.loc[X[1].isin(subj)]

    subj = X[1]

    metadata = metadata.loc[metadata["subject"].isin(subj)]

    return X , metadata

############################################
main_dir = "/home/nnieto/Nico/Harmonization/data/"
metadata_name = "meta_data"
metadata = pd.read_csv(main_dir+metadata_name)
main_dir = "/home/nnieto/Nico/Harmonization/data/extracted_data/"
############################################

sites_use = ["1000Gehirne"]
data_name = "1000brains.txt"
X_1000Gehirne, Y_1000Gehirne = get_1000_dataset(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_1000Gehirne["subject"], len(np.unique(X_1000Gehirne[1]))))
############################################

sites_use = ["CamCAN"]
data_name = "CamCAN.txt"
X_CamCAN, Y_CamCAN= get_dataset(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_CamCAN["subject"], len(np.unique(X_CamCAN[1]))))
############################################

sites_use = ['BMB_1', 'BNU_1' ,'BNU_2', 'BNU_3' ,'HNU_1',
 'IACAS' ,'IBATRT' ,'IPCAS_1' ,'IPCAS_2', 'IPCAS_3' ,'IPCAS_4',
 'IPCAS_5', 'IPCAS_6', 'IPCAS_7' ,'IPCAS_8',  'JHNU_1', 'LMU_1',
  'LMU_2', 'LMU_3','MPG_1', 'MRN_1' ,'NKI_1' ,'NYU_1', 'NYU_2',
  'SWU_1' ,'SWU_2' ,'SWU_3' ,'SWU_4' ,'UM' ,'UPSM_1',
 'UWM' ,'Utah_1', 'Utah_2', 'XHCUMS']

data_name = "CoRR.txt"
X_Corr ,  Y_Corr = get_CoRR_dataset(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_Corr["subject"], len(np.unique(X_Corr[1]))))
############################################

sites_use = ["HCP"]
data_name = "hcp.txt"
X_HCP, Y_HCP = get_dataset(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_HCP["subject"], len(np.unique(X_HCP[1]))))
############################################

sites_use = ["IXI/Guys","IXI/HH","IXI/IOP"]
data_name = "ixi.txt"
X_IXI, Y_IXI = get_dataset(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_IXI["subject"], len(np.unique(X_IXI[1]))))
############################################

sites_use = ["OASIS3"]
data_name = "OASIS3.txt"
X_OASIS3, Y_OASIS3 = get_OASIS3_dataset(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_OASIS3["subject"], len(np.unique(X_OASIS3[1]))))

############################################

sites_use = ["ID1000"]
data_name = "aomic.txt"
X_AOMIC, Y_AOMIC = get_AOMIC_1000(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_AOMIC["subject"], len(np.unique(X_AOMIC[1]))))
############################################

sites_use = ["PIOP1"]
data_name = "aomic-piop1.txt"
X_AOMIC_PIOP1, Y_AOMIC_PIOP1 = get_AOMIC_PIOP1(main_dir,sites_use,data_name,metadata)

assert len(np.unique(Y_AOMIC_PIOP1["subject"], len(np.unique(X_AOMIC_PIOP1[1]))))
############################################

sites_use = ["PIOP2"]
data_name = "aomic-piop2.txt"
X_AOMIC_PIOP2, Y_AOMIC_PIOP2 = get_AOMIC_PIOP2(main_dir,sites_use,data_name,metadata)

assert len(np.unique(Y_AOMIC_PIOP2["subject"], len(np.unique(X_AOMIC_PIOP2[1]))))
############################################

sites_use = ["eNKI"]
data_name = "eNKI.txt"
X_eNKI, Y_eNKI = get_1000_dataset(main_dir,sites_use,data_name,metadata)
assert len(np.unique(Y_eNKI["subject"], len(np.unique(X_eNKI[1]))))
############################################

X_final = pd.concat([X_1000Gehirne,X_AOMIC,X_AOMIC_PIOP1,X_AOMIC_PIOP2,
                        X_CamCAN, X_Corr,X_eNKI,X_HCP,
                        X_IXI,X_OASIS3])

Y_final = pd.concat([Y_1000Gehirne,Y_AOMIC,Y_AOMIC_PIOP1,Y_AOMIC_PIOP2,
                        Y_CamCAN, Y_Corr, Y_eNKI, Y_HCP,
                        Y_IXI, Y_OASIS3])


# Sort everything with subjects
X_final = X_final.sort_values(1)
Y_final = Y_final.sort_values("subject")

subject_Y = Y_final["subject"].reset_index()
subject_X = X_final[1].reset_index()

assert subject_Y["subject"].equals(subject_X[1])


# delete sites, subject and last column
X_final = X_final.drop([0,1,3749], axis=1)

Y_final = Y_final[["site","subject","age","gender"]]

print("1000Gehirne:" + str(len(np.unique(Y_1000Gehirne["subject"]))))
print("Corr:" + str(len(np.unique(Y_Corr["subject"]))))
print("eNKI:" + str(len(np.unique(Y_eNKI["subject"]))))
print("AOMIC_PIOP2:" + str(len(np.unique(Y_AOMIC_PIOP2["subject"]))))
print("AOMIC_PIOP1:" + str(len(np.unique(Y_AOMIC_PIOP1["subject"]))))
print("AOMIC:" + str(len(np.unique(Y_AOMIC["subject"]))))
print("OASIS3:" + str(len(np.unique(Y_OASIS3["subject"]))))
print("IXI:" + str(len(np.unique(Y_IXI["subject"]))))
print("CamCAN:" + str(len(np.unique(Y_CamCAN["subject"]))))
print("HCP:" + str(len(np.unique(Y_HCP["subject"]))))

print("All data: " + str(len(np.unique(Y_final["subject"]))))


del X_1000Gehirne,X_AOMIC,X_AOMIC_PIOP1,X_AOMIC_PIOP2, subject_Y
del X_CamCAN, X_Corr,X_eNKI,X_HCP,X_IXI,X_OASIS3, subject_X
del Y_1000Gehirne,Y_AOMIC,Y_AOMIC_PIOP1,Y_AOMIC_PIOP2
del Y_CamCAN, Y_Corr, Y_eNKI, Y_HCP, Y_IXI, Y_OASIS3
del metadata_name,  data_name, sites_use, metadata, main_dir


X_final.to_csv("/home/nnieto/Nico/Harmonization/data/X_final.csv", index=False)
Y_final.to_csv("/home/nnieto/Nico/Harmonization/data/Y_final.csv", index=False)
#%%
