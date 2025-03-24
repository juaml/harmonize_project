# %%
import pandas as pd
import numpy as np


sites = [["1000Gehirne", 60, 70],
         ["eNKI", 18, 80],
         ["CamCAN", 18, 99],
         ["ID1000", 19, 30],
         ["SALD", 18, 90],
         ["DLBS", 18, 90]]

root_dir = "/data/raw_MRI/"
save_dir = "/data/balanced/final_data_split/"
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


def filter_age_bins(Y_data, age_bins, n_images):
    bin = 0
    for t, n in enumerate(age_bins):

        if t == 0:
            age_low = n
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age*idx_age2]
            # replace the age for the next bin
            age_low = age_high
            if bin == 0:
                filter_index = Y_filt[Y_filt["gender"] == "M"].iloc[0:n_images].index                           # noqa
                filter_index = filter_index.append(Y_filt[Y_filt["gender"] == "F"].iloc[0: n_images].index)     # noqa
                bin = 1
            else:
                filter_index = filter_index.append(Y_filt[Y_filt["gender"] == "M"].iloc[0: n_images].index)     # noqa
                filter_index = filter_index.append(Y_filt[Y_filt["gender"] == "F"].iloc[0: n_images].index)     # noqa

    return filter_index


for site, low_cut_age, high_cut_age in sites:

    X_data = pd.read_csv(root_dir + "X_" + site + ".csv")
    Y_data = pd.read_csv(root_dir + "Y_" + site + ".csv")
    # Remove 18 patients
    idx_age = np.array(round(Y_data["age"]) >= low_cut_age)
    # Remove patients older than 90 years
    idx_age2 = np.array(high_cut_age >= round(Y_data["age"]))
    Y_data = Y_data[idx_age*idx_age2]

    age_min = round(Y_data["age"].min())
    age_max = round(Y_data["age"].max())
    steps = round((age_max-age_min) / n_bins)
    age_bins = range(age_min, age_max, steps)

    n_images = min_number_images(Y_data, age_bins)
    print(n_images)
    index = filter_age_bins(Y_data, age_bins, n_images)

    Y_filt = Y_data.loc[index]
    X_filt = X_data.loc[index]

    Y_filt.to_csv(save_dir + "Y_" + site + ".csv")
    X_filt.to_csv(save_dir + "X_" + site + ".csv")
    print("Done")
