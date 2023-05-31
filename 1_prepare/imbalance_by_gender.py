# %%
import pandas as pd

croping_sites = [["CamCAN", .8, .2],
                 ["eNKI", .2, .8]]

root_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split_TIV/"

for site, percent_female, percent_male in croping_sites:

    Y_data = pd.read_csv(root_dir+"Y_"+site+".csv")            # noqa
    X_data = pd.read_csv(root_dir+"X_"+site+".csv")            # noqa

    # Count the number of females and males
    num_females = Y_data['gender'].value_counts()['F']
    num_males = Y_data['gender'].value_counts()['M']

    # Calculate the number of females and males you want to keep
    num_females_to_keep = int(num_females * percent_female)
    num_males_to_keep = int(num_males * percent_male)

    # Subset the data to keep the desired number of females and males
    df1_females = Y_data[Y_data['gender'] == 'F'].sample(num_females_to_keep,
                                                         random_state=42)
    df1_males = Y_data[Y_data['gender'] == 'M'].sample(num_males_to_keep,
                                                       random_state=42)
    Y_data = pd.concat([df1_females, df1_males])

    X_data = X_data.iloc[Y_data.index, :]

    Y_data.to_csv(root_dir+"Y_"+site+"_gender_imbalance.csv", index=False) # noqa
    X_data.to_csv(root_dir+"X_"+site+"_gender_imbalance.csv", index=False) # noqa

# %%
