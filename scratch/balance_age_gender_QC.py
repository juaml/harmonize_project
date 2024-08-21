# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

sites = [
         ["eNKI", 18, 80],
         ["CamCAN", 18, 80],
         ["SALD", 18, 80]]


root_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
qc_dir = "/home/nnieto/Nico/Harmonization/data/qc/"

# save_dir = "/home/nnieto/Nico/Harmonization/data/balanced/final_data_split/"
n_bins = 10


def min_number_images(Y_data, age_bins):
    for t, n in enumerate(age_bins):
        if t == 0:
            age_low = n
            min_image = 1000
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age*idx_age2]
            # replace the age for the next bin
            age_low = age_high
            # Get the minimun value for each bin
            min_image_new = Y_filt["gender"].value_counts().min()
            if min_image_new < min_image:
                min_image = min_image_new

    return min_image


def filter_age_bins(Y_data, age_bins, n_images, sampling='random'):
    bin = 0
    filter_index = pd.Index([])  # Initialize an empty index

    for t, n in enumerate(age_bins):
        if t == 0:
            age_low = n
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age * idx_age2]

            if sampling == 'high_Q':
                # Sort by IQR in ascending order to get the lowest IQR values
                Y_filt = Y_filt.sort_values(by='IQR', ascending=True)
            elif sampling == 'low_Q':
                # Sort by IQR in descending order to get the highest IQR values
                Y_filt = Y_filt.sort_values(by='IQR', ascending=False)
            else:
                # Random sampling
                None

            # Sample n_images per gender
            males = Y_filt[Y_filt["gender"] == "M"].iloc[:n_images].index
            females = Y_filt[Y_filt["gender"] == "F"].iloc[:n_images].index

            if bin == 0:
                filter_index = males.append(females)
                bin = 1
            else:
                filter_index = filter_index.append(males)
                filter_index = filter_index.append(females)

            # Replace the age for the next bin
            age_low = age_high

    return filter_index



# %%
for site, low_cut_age, high_cut_age in sites:

    X_data = pd.read_csv(root_dir + "X_" + site + ".csv")
    Y_data = pd.read_csv(root_dir + "Y_" + site + ".csv")
    qc_data = pd.read_csv(qc_dir+site+"_cat12.8.1_rois_thalamus.csv")
    qc_data.rename(columns={"SubjectID": "subject"}, inplace=True)

    if site == "eNKI":
        qc_data = qc_data[qc_data.Session == "ses-BAS1"]
    if site == "SALD":
        qc_data['subject'] = qc_data['subject'].str.replace('sub-', '')
        qc_data['subject'] = pd.to_numeric(qc_data['subject'])

    Y_data = pd.merge(Y_data, qc_data[['subject', 'IQR']],
                      on='subject', how='left')

    # Remove under 18 patients
    idx_age = np.array(round(Y_data["age"]) >= low_cut_age)

    idx_age2 = np.array(high_cut_age >= round(Y_data["age"]))
    Y_data = Y_data[idx_age*idx_age2]

    age_min = round(Y_data["age"].min())
    age_max = round(Y_data["age"].max())
    steps = round((age_max-age_min) / n_bins)
    age_bins = range(age_min, age_max, steps)

    n_images = min_number_images(Y_data, age_bins)
    print(n_images)
    index = filter_age_bins(Y_data, age_bins, n_images, sampling="randomly")

    Y_filt = Y_data.loc[index]
    X_filt = X_data.loc[index]
    plt.figure()
    sbn.swarmplot(qc_data.IQR)

    sbn.swarmplot(Y_filt.IQR)
    print(site)
    print(Y_filt["gender"].value_counts().sum())
    print(Y_filt["gender"].value_counts())
    print("---------")
    print("Saving")
    # Y_filt.to_csv(save_dir + "Y_" + site + ".csv")
    # X_filt.to_csv(save_dir + "X_" + site + ".csv")
    print("Done")


# %%
