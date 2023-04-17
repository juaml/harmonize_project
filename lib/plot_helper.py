import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
)


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
