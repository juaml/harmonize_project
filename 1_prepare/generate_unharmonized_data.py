# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"

noise_scale = 0.05

datasets = ["eNKI", "CamCAN"]

for dataset in datasets:
    print("Loading " + dataset)
    data = pd.read_csv(root_dir + "X_"+dataset+".csv")
    Y = pd.read_csv(root_dir + "Y_"+dataset+".csv")
    # save the same targets
    Y.to_csv(root_dir+"Y_"+dataset+"_noisy.csv")
    noisy_df = pd.DataFrame()
    # Iterate over each column (feature) in the original dataframe
    for feature in data.columns:
        # Generate random mean (-1 or 1)
        random_mean = np.random.choice([-1, 1])

        # Generate random noise using the Gaussian distribution with mean and standard deviation of 1
        noise = np.random.normal(loc=random_mean, scale=noise_scale, size=len(data))

        # Add the noise to the original feature values and store in the new dataframe
        noisy_df[feature] = data[feature] + noise
    print(noisy_df.shape)
    noisy_df.to_csv(root_dir+"X_"+dataset+"_noisy.csv", index=False)
# %%

plt.scatter(data.iloc[:, 2], data.iloc[:, 3333])
plt.scatter(noisy_df.iloc[:, 2], noisy_df.iloc[:, 3333])

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(["Original features", "Noisy features"])
# %%

# %%
