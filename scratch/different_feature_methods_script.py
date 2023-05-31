# %%
import pandas as pd

root_dir = "/home/nnieto/Nico/Harmonization/data/multiple/"

exp_list = [["features_camcan_cat_icbm_BSF", "CamCan_CAT"],
            ["features_camcan_fmriprep_fsl_3_BSF", "CamCan_FSL"],
            ["features_enki_cat_icbm_BSF", "eNKI_CAT"],
            ["features_enki_fmriprep_fsl_3_BSF", "eNKI_FSL"]]

# name of the columns to extract from the data
columns = []
for n in range(1074):
    columns.append("bsf_"+str(n))

for data_name, exp_name in exp_list:
    # Read the data
    data = pd.read_csv(root_dir + data_name + ".csv")
    # Drop any missing values
    data.dropna(inplace=True)
    # Filter those patiens who QC=False
    data = data[data["QC"]]
    # Keep only 18 yo or older patients
    data = data[data["Age"] > 17]
    # Drop duplicated patients
    data.drop_duplicates(subset="name", inplace=True)
    # Keep the Age session and name of the patients
    y = data.loc[:, ["name", "Age"]]
    # Rename for a consistent naming
    y.rename(columns={"Age": "age", "name": "subject"}, inplace=True)
    # Put different experiment as different sites
    y["site"] = exp_name
    # Select the data
    X = data.loc[:, columns]
    # Saving with the experiment name
    # plt.figure()
    # sbn.swarmplot(data=y, x="age")
    print(exp_name)
    print(len(y["age"]))
    # Save
    y.to_csv(root_dir+"final_data_split/Y_"+exp_name+".csv")
    X.to_csv(root_dir+"final_data_split/X_"+exp_name+".csv")
# %%
