# %%
# Imports
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.datasets import load_mni152_template

# Direction where the data is stored (Output of shaply_to_brain.py)
root_dir = "../"
# Load the initial data
arbitrary_data = nib.load('/home/nnieto/Nico/Harmonization/harmonize_project/scratch/features_in_brain.nii') # noqa

# Load the MNI template. Resolution 8mm
mni_template = load_mni152_template(resolution=8)

# Resample image to the MNI template
nifti_image = image.resample_to_img(arbitrary_data, mni_template)

original_affine = arbitrary_data.affine
resampled_affine = nifti_image.affine

# Get the affine matrix
affine_matrix = nifti_image.affine

# Initialize lists to store voxel coordinates and features
voxel_coordinates_list = []
features_list = []

# Iterate through each voxel and get its coordinates and feature
for x in range(nifti_image.shape[0]):
    for y in range(nifti_image.shape[1]):
        for z in range(nifti_image.shape[2]):
            # Voxel coordinates in the image
            voxel_coor = [x, y, z]

            # Convert voxel coordinates to real-world coordinates (x, y, z)
            real_world_coordinates = nib.affines.apply_affine(affine_matrix,
                                                              voxel_coor)

            # Extract x, y, and z coordinates
            x_coord, y_coord, z_coord = real_world_coordinates

            # Get the feature value for the current voxel
            feature_value = nifti_image.get_fdata()[x, y, z]

            # Append the coordinates and feature value to the respective lists
            voxel_coordinates_list.append([x_coord, y_coord, z_coord])
            features_list.append(feature_value)

# Create a DataFrame to store the data
data_df = pd.DataFrame({'X': [coord[0] for coord in voxel_coordinates_list],
                        'Y': [coord[1] for coord in voxel_coordinates_list],
                        'Z': [coord[2] for coord in voxel_coordinates_list],
                        'Feature': features_list})

# Delete those features that are 0
data_df = data_df[data_df["Feature"] != 0]

# Save the DataFrame to a CSV file
data_df.to_csv('coordinates_and_features.csv', index=False)
