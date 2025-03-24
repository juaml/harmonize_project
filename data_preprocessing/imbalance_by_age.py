# %%
import pandas as pd

# Define the cropping parameters for each site
cropping_sites = [
    ["1000Gehirne", 61, 80],
    ["CamCAN", 41, 60],
    ["eNKI", 27, 40]
]

# Process each site
for site, min_age, max_age in cropping_sites:
    # Load the data for the current site
    Y_data = pd.read_csv(f"/data/raw_MRI/Y_{site}.csv")  # Load target data
    X_data = pd.read_csv(f"/data/raw_MRI/X_{site}.csv")  # Load feature data

    # Filter rows based on the age range
    age_filter = (Y_data["age"] >= min_age) & (Y_data["age"] <= max_age)
    Y_data = Y_data[age_filter]
    X_data = X_data[age_filter]

    # Save the cropped data to new files
    Y_data.to_csv(f"/MRI/dependence/Y_{site}_crop.csv", index=False)
    X_data.to_csv(f"/MRI/dependence/X_{site}_crop.csv", index=False)
