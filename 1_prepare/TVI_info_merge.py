#%%

import numpy as np
import pandas as pd



def put_TIV_in_array(TIV_data,Y):

    
    dataset = np.unique(Y["site"])

    # keep only the data for the dataset
    if dataset == "ID1000":
        TIV_data = TIV_data[TIV_data[0]=="aomic_cat12.8_1mm"]
    elif dataset == "1000Gehirne": 
        TIV_data = TIV_data[TIV_data[0]=="1000brains_cat12.8_1mm"]
    elif dataset == "CamCAN": 
        TIV_data = TIV_data[TIV_data[0]=="CamCAN_cat12.8_1mm"]
    elif dataset == "eNKI": 
        TIV_data = TIV_data[TIV_data[0]=="eNKI_cat12.8_1mm"]

    
    # extract the presents subjects
    subjects_tiv = np.unique(TIV_data[1])
    sub_Y = np.unique(Y["subject"])


    for sub in subjects_tiv:

        if dataset == "ID1000":
            sub_orig = sub
            sub = sub + "_site-ID1000"
            
        # Check if the subject is in the Y data
        if sub in sub_Y:
            # If so, check where
            where_index = np.where(Y["subject"]==sub)
            # Extract the TIV data

            # back to the original name
            if dataset == "ID1000":
                sub = sub_orig 

            TIV = TIV_data[TIV_data[1]==sub][3].to_numpy()
            # Put the TIV data in the Y 
            Y.iloc[where_index[0][0],4] = TIV[0]

    return Y


def remove_extreme_TIV(X,Y,TIV_percentage):

    # Get each gender
    male = Y[Y["gender"]=="M"]
    female = Y[Y["gender"]=="F"]

    # Select males
    num_to_delete = male.shape[0] - np.round(male.shape[0] * TIV_percentage / 100).astype(int)

    gender_TIV = male["TIV"]

    sort_index = np.argsort(gender_TIV.to_numpy())

    TIV_to_remove = male.iloc[sort_index[num_to_delete],4]

    mask = np.where(gender_TIV>TIV_to_remove)

    male.reset_index(inplace=True)

    males_to_keep = male.drop(mask[0])

    # select females
    num_to_delete =  np.round(female.shape[0] * TIV_percentage / 100).astype(int)

    gender_TIV = female["TIV"]

    sort_index = np.argsort(gender_TIV.to_numpy())

    TIV_to_remove = female.iloc[sort_index[num_to_delete],4]

    mask = np.where(gender_TIV<TIV_to_remove)

    female.reset_index(inplace=True)
    females_to_keep = female.drop(mask[0])

    index_to_keep = pd.concat([males_to_keep["index"],females_to_keep["index"]])

    y_tvi = Y.drop(index_to_keep)

    Y_final = Y.drop(y_tvi.index)
    X_final = X.drop(y_tvi.index)

    return X_final, Y_final

def filter_no_TIV_info(X,Y):
    pos = Y[Y['TIV'].isna()]


    if not (len(pos)==0):
        Y = Y.drop(pos.index[0])

        X = X.drop(pos.index[0])

    return X , Y


#### Processing
TIV = pd.read_csv("/home/nnieto/Nico/Harmonization/data/TIV.csv",header=None)

sites = ["CamCAN", "ID1000", "1000Gehirne", "eNKI"]

import matplotlib.pyplot as plt

for site in sites:
    TIV_percentage = 20
    Y = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_"+site+".csv")
    X = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_"+site+".csv")

    # Initialize  TIV column 
    Y["TIV"] = -1
    # Add TIV information
    Y = put_TIV_in_array(TIV,Y)
    # Filter particiants without TIV info
    X , Y = filter_no_TIV_info(X,Y)
    # Save new files
    #X.to_csv("/home/nnieto/Nico/Harmonization/data/final_data_split_TIV/X_"+site+".csv", index=False)
    #Y.to_csv("/home/nnieto/Nico/Harmonization/data/final_data_split_TIV/Y_"+site+".csv", index=False)
    X , Y = remove_extreme_TIV(X,Y,TIV_percentage)
    plt.figure()
    male = Y[Y["gender"]=="M"]
    female = Y[Y["gender"]=="F"]
    plt.scatter(male["age"],male["TIV"],c="blue")
    plt.scatter(female["age"],female["TIV"], c="red")
    plt.xlabel("Age")
    plt.ylabel("TIV")
    plt.title(site)
    plt.show()
# %%


