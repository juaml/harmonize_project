# %%
import pandas as pd
import numpy as np

# Load data
root_dir = "/data/ADNI_raw/"
save_dir = "/data/ADNI/"
data = pd.read_csv(root_dir + "adni-aibl_dataset.csv")

# Keep only images adquired with 3T
data = data[data["meta.Imaging.Protocol.FieldStrength"] == 3]

# remove site with few images
min_site_images = 30
sites_coount = data["site"].value_counts()

selected_sites = sites_coount[sites_coount >= min_site_images].index
data_filtered = data[data['site'].isin(selected_sites)]

print(data_filtered['site'].value_counts())

# %%
# Replace values with and colum names
Y_ANDI = data_filtered.loc[:, ["id", "age", "groupHarmonized", "site"]]

# To use the same scripts for sex classification
Y_ANDI["groupHarmonized"].replace({"dementia": "M",
                                   "mci": "M", "control": "F"},
                                  inplace=True)


# put grup as gender for classification and script reuse
Y_ANDI.rename(columns={"groupHarmonized": "gender"}, inplace=True)

# %%
# Preprocess data
X_ANDI = data_filtered.filter(regex="fs", axis=1)
X_ANDI = X_ANDI.iloc[:, 1::]

# delete low variance features
colvar = np.var(X_ANDI, axis=0)
# Identify variables with low variance
low_var_threshold = 1e-5
high_var_threshold = 5
# Filter columns based on variance thresholds

# Filter columns based on variance thresholds
low_var_cols = colvar[colvar < low_var_threshold].index
high_var_cols = colvar[colvar > high_var_threshold].index

# Create new filtered DataFrame
X_ANDI = X_ANDI.drop(columns=np.concatenate([low_var_cols, high_var_cols]))

# Create new filtered matrix
# %%
for site in np.unique(Y_ANDI["site"]):
    y_site = Y_ANDI[Y_ANDI["site"] == site]
    x_site = X_ANDI[Y_ANDI["site"] == site]
    Y_ANDI["site"].replace({site: "SITE_"+str(site)}, inplace=True)

    y_site.to_csv(save_dir + "Y_SITE_" + str(site) + ".csv")
    x_site.to_csv(save_dir + "X_SITE_" + str(site) + ".csv")

# %%
Y_ANDI.to_csv(save_dir + "Y_Kersten.csv")
X_ANDI.to_csv(save_dir + "X_Kersten.csv")

# Need to split the datasets in the sites
# %%
print("Images by Sites ")
print(Y_ANDI["site"].value_counts())
print("Diagnosis count")
print(Y_ANDI["gender"].value_counts())
# %%
