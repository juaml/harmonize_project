import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import nibabel.processing as npr
from nilearn import plotting


def plot_regression(data, harm_modes, absolute):
    data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                           "target": "Leakage",
                                           "none": "None",
                                           "cheat": "Cheat"}, inplace=True)

    if absolute:
        data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
    else:
        data["y_diff"] = data["y_true"]-data["y_pred"]

    sbn.catplot(
        data=data, kind="boxen",
        x="site", y="y_diff", hue="Harmonization Schemes",
        height=8, hue_order=harm_modes, legend_out=False
    )
    plt.ylabel("Age prediction difference [years]")
    plt.title("Harmonization Schemes")
    plt.title("Age Prediction")
    plt.grid(alpha=0.5, axis="y", c="black")
    plt.show()
    return


def plot_oos_regression(data, harm_modes, absolute):

    data["Harmonization Schemes"].replace(
                                {"pretend_nosite": "JuHarmonize No site",
                                 "none": "None"},
                                inplace=True)

    # Select methods to plot
    data = data[data["Harmonization Schemes"].isin(harm_modes)]

    # Plot
    _, ax = plt.subplots(1, 1, figsize=[20, 10])

    if absolute:
        data["y_diff"] = np.abs(data["y_true"] -
                                data["y_pred"])
    else:
        data["y_diff"] = data["y_true"]-data["y_pred"]

    sbn.swarmplot(
        data=data,
        x="site", y="y_diff", hue="Harmonization Schemes",
        hue_order=harm_modes, dodge=True, ax=ax
    )
    sbn.boxplot(
        data=data, color="w", zorder=1,
        x="site", y="y_diff", hue="Harmonization Schemes",
        hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])

    plt.ylabel("Age Difference")
    plt.xlabel("Sites ID")
    plt.title("Age regression - OOS Experiment")
    plt.grid(alpha=0.5, axis="y", c="black")
    plt.show()
    return


def plot_oos_classification(data, harm_modes):
    data["Harmonization Schemes"].replace({
                                    "pretend_nosite": "JuHarmonize No site",
                                    "none": "None"},
                                    inplace=True)

    # Select methods to plot
    data = data[
                 data["Harmonization Schemes"].isin(harm_modes)]
    # Plot

    # Plot
    _, ax = plt.subplots(1, 1, figsize=[20, 10])

    sbn.barplot(
        data=data, zorder=1,
        x="site", y="balance_acc", hue="Harmonization Schemes",
        hue_order=harm_modes, dodge=True, ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])
    ax.axhline(0.5, lw=2, color="k", ls="--", alpha=1, label="Chance level")

    plt.ylabel("Balanced Accuracy")
    plt.xlabel("Sites ID")

    # ax.set_xticklabels(site_list)
    plt.title("Gender Classification - OOS Experiment")
    plt.grid(alpha=0.5, axis="y", c="black")
    plt.show()
    return


def plot_classification(data, harm_modes, site_order):
    # data to appropiated names
    data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                          "target": "Leakage",
                                           "none": "None",
                                           "cheat": "Cheat"},
                                          inplace=True)

    # Select methods to plot
    data = data[data["Harmonization Schemes"].isin(harm_modes)]

    # Plot
    _, ax = plt.subplots(1, 1, figsize=[20, 10])
    sbn.swarmplot(
        data=data,
        x="site", y="auc", hue="Harmonization Schemes",
        order=site_order,
        hue_order=harm_modes, dodge=True, ax=ax
    )
    sbn.boxplot(
        data=data, color="w", zorder=1,
        x="site", y="auc", hue="Harmonization Schemes",
        order=site_order,
        hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])
    plt.ylabel("AUC")
    plt.xlabel("Sites")
    plt.title("Gender Classification")
    plt.grid(alpha=0.5, axis="y", c="black")
    plt.show()
    return


def binarize_3d(img, threshold):
    """binarize 3D spatial image"""
    return nib.Nifti1Image(np.where(img.get_fdata() > threshold, 1, 0),
                           img.affine, img.header)


def map_values_to_brian(mask_dir, values, threshold,
                        resample_size_maks, save_dir=None, colorbar=True):
    # Load tha mask
    mask = nib.load(mask_dir)
    # Reshape the mask
    mask_img_rs = npr.resample_to_output(mask, [resample_size_maks] * len(mask.shape), order=1)  # noqa
    # Binarize with a fix threshold
    mask_rs = binarize_3d(mask_img_rs, 0.5)
    # get the data
    mask_rs = mask_rs.get_fdata()
    # Make sure that the mask have as many points as the features
    assert (mask_rs.sum() == values.shape[0])
    # flatten the mask to match the features
    mask_rsample_rshape = mask_rs.reshape(-1, 1)
    # Start replacing the mask with the values
    position = 0
    for index, num in enumerate(mask_rsample_rshape):
        # if a 1 is found in the mask, replace the value
        if num == 1:
            # replace the value and update the position
            mask_rsample_rshape[index][0] = values[position]
            position = position + 1

    # Make sure that all the features were used
    assert (position == values.shape[0])
    # Reshape back the mask in a volume
    mask_weights = mask_rsample_rshape.reshape(mask_img_rs.shape)
    # Create a Nifti image
    mask_weights = nib.Nifti1Image(mask_weights, mask_img_rs.affine, mask_img_rs.header)      # noqa
    # Create a glass plot with the passed threshold
    plotting.plot_glass_brain(mask_weights, threshold=threshold,
                              colorbar=colorbar)

    if save_dir is not None:
        nib.save(mask_weights, 'features_in_brain.nii')
    return
