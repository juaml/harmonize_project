# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

sites = [["1000Gehirne", 60, 70],
         ["eNKI", 18, 80],
         ["CamCAN", 18, 99],
         ["ID1000", 19, 30],
         ["SALD", 18, 90],
         ["DLBS", 18, 90]]

# sites = [["SALD",18,90],
#          ["DLBS",20,90]]

root_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
save_dir = "/home/nnieto/Nico/Harmonization/data/balanced/final_data_split/"
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
                filter_index = Y_filt[Y_filt["gender"] == "M"].iloc[0:n_images].index
                filter_index = filter_index.append(Y_filt[Y_filt["gender"] == "F"].iloc[0: n_images].index)
                bin = 1
            else:
                filter_index = filter_index.append(Y_filt[Y_filt["gender"] == "M"].iloc[0: n_images].index)
                filter_index = filter_index.append(Y_filt[Y_filt["gender"] == "F"].iloc[0: n_images].index)

    return filter_index


for site, low_cut_age, high_cut_age in sites:

    X_data = pd.read_csv(root_dir + "X_" + site + ".csv")
    Y_data = pd.read_csv(root_dir + "Y_" + site + ".csv")
    # Remove 18 patients
    idx_age = np.array(round(Y_data["age"]) >= low_cut_age)

    idx_age2 = np.array(high_cut_age >= round(Y_data["age"]))
    Y_data = Y_data[idx_age*idx_age2]
    # Y_data = Y_data[idx_age]
    # X_data = X_data[idx_age]
    age_min = round(Y_data["age"].min())
    age_max = round(Y_data["age"].max())
    steps = round((age_max-age_min) / n_bins)
    age_bins = range(age_min, age_max, steps)

    n_images = min_number_images(Y_data, age_bins)
    print(n_images)
    index = filter_age_bins(Y_data, age_bins, n_images)

    Y_filt = Y_data.loc[index]
    X_filt = X_data.loc[index]
    # plt.figure()
    # male = Y_filt[Y_filt["gender"] == "M"]
    # female = Y_filt[Y_filt["gender"] == "F"]
    # plt.scatter(female["age"], female["TIV"], c="red")
    # plt.scatter(male["age"], male["TIV"], c="blue")
    # plt.xlabel("Age")
    # plt.ylabel("TIV")
    # plt.title(site)
    # plt.show()

    print(site)
    print(Y_filt["gender"].value_counts().sum())
    print(Y_filt["gender"].value_counts())
    print("---------")
    print("Saving")
    # Y_filt.to_csv(save_dir + "Y_" + site + ".csv")
    # X_filt.to_csv(save_dir + "X_" + site + ".csv")
    print("Done")

# # %%
# import pandas as pd
# y_SALD = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_SALD.csv")
# y_DLBS = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_DLBS.csv")

# # %%
# import pandas as pd
# y_DLBS = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_DLBS.csv")
# y_DLBS["site"] = "DLBS_full"
# X_DLBS = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_DLBS.csv")

# # %%
# save_dir = "/home/nnieto/Nico/Harmonization/data/balanced/final_data_split/"
# y_DLBS.to_csv(save_dir + "Y_DLBS_full.csv")
# X_DLBS.to_csv(save_dir + "X_DLBS_full.csv")
# %%
