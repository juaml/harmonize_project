# %%
import pandas as pd


def process_site(site, percent_female, percent_male, root_dir):
    # Load data
    Y_data = pd.read_csv(root_dir + "Y_" + site + ".csv")
    X_data = pd.read_csv(root_dir + "X_" + site + ".csv")

    # Count the number of females and males
    num_females = Y_data['gender'].value_counts().get('F', 0)
    num_males = Y_data['gender'].value_counts().get('M', 0)

    # Calculate the number of females and males you want to keep
    num_females_to_keep = int(num_females * percent_female)
    num_males_to_keep = int(num_males * percent_male)

    # Subset the data to keep the desired number of females and males
    df1_females = Y_data[Y_data['gender'] == 'F'].sample(num_females_to_keep,
                                                         random_state=42)
    df1_males = Y_data[Y_data['gender'] == 'M'].sample(num_males_to_keep,
                                                       random_state=42)
    Y_data_filtered = pd.concat([df1_females, df1_males])

    # Ensure X_data has the same number of samples as Y_data_filtered
    X_data_filtered = X_data.loc[Y_data_filtered.index, :]

    return Y_data_filtered, X_data_filtered


def balance_datasets(root_dir, croping_sites):
    # Process each site and get filtered datasets
    datasets = [process_site(site, percent_female, percent_male, root_dir) for site, percent_female, percent_male in croping_sites]         # noqa

    # Determine the minimum length among the datasets
    min_length = min(len(Y_data) for Y_data, _ in datasets)

    # Downsample each dataset to the minimum length
    balanced_datasets = [(Y_data.sample(n=min_length, random_state=42),
                          X_data.loc[Y_data.sample(n=min_length, random_state=42).index, :]) for Y_data, X_data in datasets]                # noqa

    # Save the balanced datasets
    for (Y_data, X_data), (site, _, _) in zip(balanced_datasets, croping_sites):        # noqa
        print(site)
        print(Y_data["gender"].value_counts())
        Y_data.to_csv(root_dir + "Y_" + site + "_gender_imbalance_extreme.csv",
                      index=False)
        X_data.to_csv(root_dir + "X_" + site + "_gender_imbalance_extreme.csv",
                      index=False)


# Parameters
croping_sites = [["CamCAN", 0.95, 0.05],
                 ["eNKI", 0.05, 0.95]]
root_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"

# Run the balancing function
balance_datasets(root_dir, croping_sites)

# %%
