# %%
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

# %%
croping_sites = [["1000Gehirne", 61, 80],
                 ["CamCAN", 41, 60],
                 ["eNKI", 27, 40]]


for site, min_age, max_age in croping_sites:

    Y_data = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_"+site+".csv")            # noqa
    X_data = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_"+site+".csv")            # noqa
    index = Y_data["age"] >= min_age
    Y_data = Y_data[index]
    X_data = X_data[index]
    index = max_age >= Y_data["age"]
    Y_data = Y_data[index]
    X_data = X_data[index]
    Y_data.to_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_"+site+"_crop.csv", index=False) # noqa
    X_data.to_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_"+site+"_crop.csv", index=False) # noqa

# %%

Y_1000brains = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_1000Gehirne_crop.csv")  # noqa
Y_Camcan = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_CamCAN_crop.csv")           # noqa
Y_ID1000 = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_ID1000.csv")                # noqa
Y_eNKI = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_eNKI_crop.csv")               # noqa
Y_SALD = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_SALD_crop.csv")               # noqa

data = pd.concat([Y_1000brains, Y_Camcan, Y_eNKI, Y_SALD, Y_ID1000])
fig = plt.figure(figsize=[20, 10])

ax = fig.add_subplot(1, 1, 1)
sbn.swarmplot(data=data, x="age", y="site", ax=ax, hue="gender", size=4)

# %%
Y_data = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_SALD.csv")            # noqa
sbn.swarmplot(data=Y_data, x="age", y="site")

# %%
Y_1000brains = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_1000Gehirne.csv")  # noqa
Y_Camcan = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_CamCAN.csv")           # noqa
Y_ID1000 = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_ID1000.csv")                # noqa
Y_eNKI = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_eNKI.csv")               # noqa
Y_SALD = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_SALD.csv")               # noqa

data = pd.concat([Y_1000brains, Y_Camcan, Y_eNKI, Y_SALD, Y_ID1000])
fig = plt.figure(figsize=[20, 10])

ax = fig.add_subplot(1, 1, 1)
sbn.swarmplot(data=data, x="age", y="site", ax=ax, hue="gender")
# %%
