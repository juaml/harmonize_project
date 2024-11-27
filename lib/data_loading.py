import pandas as pd
import os
from lib.data_processing import balance_gender, retain_images


def load_ADNI(data_dir, site_target_independance=True, random_state=23):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname((__file__))))            # noqa

    X1 = pd.read_csv(project_root+data_dir+"X_SITE_1.csv")
    X2 = pd.read_csv(project_root+data_dir+"X_SITE_2.csv")
    Y1 = pd.read_csv(project_root+data_dir+"Y_SITE_1.csv")
    Y2 = pd.read_csv(project_root+data_dir+"Y_SITE_2.csv")

    if site_target_independance:
        # Same number of samples for male and female in each site
        n_S1_female = 126
        n_S1_male = 126
        n_S2_male = 57
        n_S2_female = 57
    else:
        n_S1_female = 10
        n_S1_male = 100
        n_S2_male = 10
        n_S2_female = 100

    # Generate dependance
    # Select 100 "F" and 1 "M" from Y1
    Y1_females = Y1[Y1['gender'] == 'F'].sample(n=n_S1_female,
                                                random_state=random_state)
    Y1_male = Y1[Y1['gender'] == 'M'].sample(n=n_S1_male,
                                             random_state=random_state)

    # Combine the selected "F" and "M" to form the new Y1
    new_Y1 = pd.concat([Y1_females, Y1_male])

    # Ensure X1 is synchronized with the
    # new Y1 by selecting the corresponding rows
    new_X1 = X1.loc[new_Y1.index]

    # Select 100 "M" and 1 "F" from Y2
    Y2_males = Y2[Y2['gender'] == 'M'].sample(n=n_S2_male,
                                              random_state=random_state)
    Y2_female = Y2[Y2['gender'] == 'F'].sample(n=n_S2_female,
                                               random_state=random_state)

    # Combine the selected "M" and "F" to form the new Y2
    new_Y2 = pd.concat([Y2_males, Y2_female])

    # Ensure X2 is synchronized with the new Y2
    # by selecting the corresponding rows
    new_X2 = X2.loc[new_Y2.index]

    X = pd.concat([new_X1, new_X2])
    Y = pd.concat([new_Y1, new_Y2])

    X = X.to_numpy()
    sites = Y["site"].reset_index()
    Y = Y["gender"].replace({"F": 1, "M": 0}).to_numpy()
    return X, Y, sites


def load_crop_dataset(name, data_dir):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))               # noqa

    Y = pd.read_csv(project_root+data_dir+"Y_"+name+"_crop.csv")
    X = pd.read_csv(project_root+data_dir+"X_"+name+"_crop.csv")
    return X, Y


def load_sex_age_balanced_data(data_dir):
    X_SALD, Y_SALD = load_balanced_dataset("SALD", data_dir)
    X_eNKI, Y_eNKI = load_balanced_dataset("eNKI", data_dir)
    X_Camcan, Y_Camcan = load_balanced_dataset("CamCAN", data_dir)

    # Y_SALD = balance_gender(Y_SALD, min_images)
    # Y_eNKI = balance_gender(Y_eNKI, min_images)
    # Y_Camcan = balance_gender(Y_Camcan, min_images)

    Y = pd.concat([Y_SALD, Y_eNKI, Y_Camcan])

    X = pd.concat([X_SALD, X_eNKI, X_Camcan])

    X.dropna(axis=1, inplace=True)
    Y["site"].replace({"SALD": 0, "eNKI": 1,
                       "CamCAN": 2}, inplace=True)
    sites = Y["site"].reset_index()
    Y["gender"].replace({"F": 0, "M": 1}, inplace=True)

    Y = Y["age"].to_numpy()
    X = X.to_numpy()

    return X, Y, sites


def load_balanced_dataset(name, data_dir):
    data_folder_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), 'data')
    Y = pd.read_csv(data_folder_path+data_dir+"Y_"+name+".csv", index_col=0)
    X = pd.read_csv(data_folder_path+data_dir+"X_"+name+".csv", index_col=0)
    return X, Y


def load_sex_age_imbalanced_data(data_dir, min_images):

    X_ID1000, Y_ID1000 = load_crop_dataset("ID1000", data_dir)
    X_eNKI, Y_eNKI = load_crop_dataset("eNKI", data_dir)
    X_Camcan, Y_Camcan = load_crop_dataset("CamCAN", data_dir)
    X_1000brains, Y_1000brains = load_crop_dataset("1000Gehirne", data_dir)

    Y_ID1000 = balance_gender(Y_ID1000, min_images)
    Y_eNKI = balance_gender(Y_eNKI, min_images)
    Y_Camcan = balance_gender(Y_Camcan, min_images)
    Y_1000brains = balance_gender(Y_1000brains, min_images)

    Y = pd.concat([Y_ID1000, Y_eNKI, Y_Camcan, Y_1000brains])

    X = pd.concat([retain_images(X_ID1000, Y_ID1000),
                  retain_images(X_eNKI, Y_eNKI),
                  retain_images(X_Camcan, Y_Camcan),
                  retain_images(X_1000brains, Y_1000brains)])

    X.dropna(axis=1, inplace=True)
    Y["site"].replace({"SALD": 0, "eNKI": 1,
                       "CamCAN": 2}, inplace=True)
    Y["site"].replace({"ID1000": 0, "eNKI": 1,
                       "CamCAN": 2, "1000Gehirne": 3}, inplace=True)
    Y["gender"].replace({"F": 0, "M": 1}, inplace=True)
    sites = Y["site"].reset_index()

    Y = Y["age"].to_numpy()
    X = X.to_numpy()

    return X, Y, sites


def load_eICU(data_dir):
    # Load data (This data was obtained from the eICU dataset)
    # please contact the authors for obtaining the subject ids to replicate
    project_root = os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))             # noqa
    data = pd.read_csv(project_root+data_dir + "equals_to_paper_data.csv",
                       index_col=0)

    return data


def load_MRI_sex_clf_site_target_dependance(data_dir):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa
    root_dir = project_root + data_dir
    data_enki = pd.read_csv(root_dir+"X_eNKI_gender_imbalance_extreme.csv")
    data_CamCAN = pd.read_csv(root_dir+"X_CamCAN_gender_imbalance_extreme.csv")

    y_enki = pd.read_csv(root_dir+"Y_eNKI_gender_imbalance_extreme.csv")
    y_enki["site"] = "eNKI"
    y_CamCAN = pd.read_csv(root_dir+"Y_CamCAN_gender_imbalance_extreme.csv")
    y_CamCAN["site"] = "CamCAN"

    X = pd.concat([data_CamCAN, data_enki])
    X.dropna(axis=1, inplace=True)
    X = X.to_numpy()
    target = pd.concat([y_CamCAN, y_enki])

    target["site"].replace({"eNKI": 0, "CamCAN": 1}, inplace=True)
    sites = target["site"].reset_index()
    target["gender"].replace({"F": 0, "M": 1}, inplace=True)

    Y = target["gender"]

    return X, Y, sites
