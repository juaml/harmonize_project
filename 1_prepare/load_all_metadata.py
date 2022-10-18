###### IMPORTANT
# The code is made to run in the ADNI folder. 
# The files Diagnosis and MR_Image_Analysis_ADNI_1_2_GO_FS51 should be unzip
# The final data is not stored. 
#
# Final data shape
# data = contains only the TA information   - ndarray [participants, features]
# Gender = Contains the patients gender     - ndarray [participants,]
# Age = Contains the age of the patients    - ndarray [participants,]
# Site = Contains the site information      - ndarray [participants,]

# %%

### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

def load_AOMIC_metadata(data_dir, load_dataset="all"):

    if load_dataset == "all":
        load_dataset = ["PIOP1","PIOP2","ID1000"]
    
    for n,d in enumerate(load_dataset):

        data_name = "AOMIC/AOMIC-"+d+"_participants.tsv"

        data = pd.read_csv(data_dir+data_name,sep='\t')
    
        data = data[["participant_id","age","sex"]]

        data["site"] = d

        for nn,sub in enumerate(data["participant_id"]):
            data["participant_id"][nn] = sub + "_site-" + d


        assert len(np.unique(data["participant_id"].to_numpy()) ) == data.shape[0]

        if n== 0:
            data_final = data
        else:
            data_final = pd.concat([data_final,data])

    # Drop some participants with no information
    data_final = data_final.dropna()

    # Concistent naming
    data_final["sex"].replace({"female" : "F", "male": "M"},inplace=True)
 
    data_final.rename(columns={"sex": "gender"}, inplace=True)
    data_final.rename(columns={"participant_id": "subject"}, inplace=True)
    assert len(np.unique(data_final["subject"].to_numpy()) ) == data_final.shape[0]

    return data_final

data_dir = "/home/nnieto/Nico/Harmonization/datasets_metadata/"
data_AOMIC = load_AOMIC_metadata(data_dir)

def  load_dataset(data_dir, file_name):

    data = pd.read_csv(data_dir+file_name)

    data = data[["site","subject","age","gender"]]

    data = data.dropna()

    assert len(np.unique(data["subject"].to_numpy()) ) == data.shape[0]
    return data
file_name = "1000brains_subject_list_withTIV.csv"
data_1000 = load_dataset(data_dir, file_name)

file_name = "camcan_subject_list_withTIV.csv"
data_camcan = load_dataset(data_dir, file_name)

def  load_eNKI(data_dir, file_name):

    data = pd.read_csv(data_dir+file_name, sep=";")

    data = data[["sample/site","participant_id","age","sex"]]

    data = data.dropna()



    data.rename(columns={"sample/site": "site"}, inplace=True)
    data.rename(columns={"sex": "gender"}, inplace=True)
    data.rename(columns={"participant_id": "subject"}, inplace=True)

        # Keep only the oldes image for each RID
    for num, RID in enumerate(np.unique(data["subject"])):
        data_aux = data[data["subject"]==RID]

        id_min = np.min(data_aux["age"])

        data_aux = data_aux[data_aux["age"]==id_min].iloc[0]

        if num == 0:
            data_return = data_aux
        else:
            data_return = np.vstack([data_return,data_aux])

    data = pd.DataFrame(data_return, columns = data.keys())


    assert len(np.unique(data["subject"].to_numpy()) ) == data.shape[0]
    return data

data_dir = "/home/nnieto/Nico/Harmonization/datasets_metadata/"
file_name = "eNKI_participants_list.csv"
data_enki = load_eNKI(data_dir, file_name)


file_name = "ixi_subject_list_withTIV.csv"
data_IXI = load_dataset(data_dir, file_name)

# Add site to subject name
for n,sub in enumerate(data_IXI["subject"]):
    data_IXI["subject"][n] = sub + "_site-" + data_IXI["site"][n].split("/",1)[1].upper()

def  load_OASIS(data_dir, file_name):

    data = pd.read_csv(data_dir+file_name)

    data = data[["Subject","Age","M/F","Scanner"]]
    data = data[data["Scanner"]=="3.0T"]
    data = data[["Subject","Age","M/F"]]
    data["site"] = "OASIS3"

    data.rename(columns={"M/F": "gender"}, inplace=True)
    data.rename(columns={"Subject": "subject"}, inplace=True)
    data.rename(columns={"Age": "age"}, inplace=True)
    
    # Keep only the oldes image for each RID
    for num, RID in enumerate(np.unique(data["subject"])):
        data_aux = data[data["subject"]==RID]

        id_min = np.min(data_aux["age"])

        data_aux = data_aux[data_aux["age"]==id_min].iloc[0]

        if num == 0:
            data_return = data_aux
        else:
            data_return = np.vstack([data_return,data_aux])

    data = pd.DataFrame(data_return, columns = data.keys())



    assert len(np.unique(data["subject"].to_numpy())) == data.shape[0]

    return data
file_name = "FelixH_4_25_2022_8_46_56.csv"
data_OASIS3 = load_OASIS(data_dir, file_name)

def  load_HCP(data_dir, file_name):

    data = pd.read_csv(data_dir+file_name)

    data.rename(columns={"Subject": "subject"}, inplace=True)
    data.rename(columns={"Age_in_Yrs": "age"}, inplace=True)
    data.rename(columns={"Gender": "gender"}, inplace=True)

    data = data[["subject","age","gender"]]
    data["site"] = "HCP"
    data['gender'] = data['gender'].replace({1: 'M', 2: 'F'})

    data = data.dropna()
    assert len(np.unique(data["subject"].to_numpy()) ) == data.shape[0]

    return data
file_name = "covariates_age_sex_ICV_HCP_n399.csv"
data_HCP = load_HCP(data_dir, file_name)

for nn,sub in enumerate(data_HCP["subject"]):
    data_HCP["subject"][nn] = "sub-" + str(data_HCP["subject"][nn])


def  load_CORR(data_dir, file_name):

    data = pd.read_csv(data_dir+file_name)

    data = data[["SITE","SUBID","AGE_AT_SCAN_1","SEX","SESSION"]]

    data = data[data["SESSION"]=="Baseline"].reset_index()

    data = data[["SITE","SUBID","AGE_AT_SCAN_1","SEX"]]

    data.rename(columns={"SITE": "site"}, inplace=True)
    data.rename(columns={"SEX": "gender"}, inplace=True)
    data.rename(columns={"SUBID": "subject"}, inplace=True)
    data.rename(columns={"AGE_AT_SCAN_1": "age"}, inplace=True)

    data["gender"].replace({"1":"F", "2": "M"},inplace=True)
   
    for nn,_ in enumerate(data["subject"]):
        data["subject"][nn] = str(data["subject"][nn]).zfill(7)

    data = data.dropna()

    assert len(np.unique(data["subject"].to_numpy()) ) == data.shape[0]
    return data

file_name = "CoRR_AggregatedPhenotypicData.csv"

data_corr= load_CORR(data_dir, file_name)

print(len(np.unique(data_corr["site"])))
print((np.unique(data_corr["site"])))

del file_name, data_dir, sub, n, nn


data_final = pd.concat([data_1000,data_AOMIC,data_camcan,
                        data_corr,data_enki,data_HCP,
                        data_IXI,data_OASIS3])

print("1000Gehirne: "+ str(data_1000.shape[0]))
print("Full AOMIC: "+str(data_AOMIC.shape[0]))
print("CAMCAN: " + str(data_camcan.shape[0]))
print("OASIS3: " + str(data_OASIS3.shape[0]))
print("eNKI: "+ str(data_enki.shape[0]))
print("HCP: " + str(data_HCP.shape[0]))
print("CoRR: " + str(data_corr.shape[0]))
print("IXI: " + str(data_IXI.shape[0]))

print("Final dataset: " + str(data_final.shape[0]))


data_final.to_csv("/home/nnieto/Nico/Harmonization/data/meta_data")


# %%

