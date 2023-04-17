# %%
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

# Load predictions
data_dir = "/home/nnieto/Nico/Harmonization/result_regression/results_regression/test_regression_all_big_rf_stack_rvr_pred/"
resutls = pd.read_csv(data_dir + "pretend_fold_0_of_5_out.csv",
                      sep=";")
resutls.rename(columns={"y_true": "True Age", "y_pred": "Predicted Age"},
               inplace=True)

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# print configuration
fig, ax = plt.subplots(1, 1, figsize=[15, 10])

# sbn.scatterplot(data=resutls, x="True Age", y="Predicted Age", hue="site",
#                 ax=ax)
sbn.regplot(data=resutls, x="True Age", y="Predicted Age", ci=100, x_jitter=0,
            scatter_kws={"s": 10}, ax=ax,
            line_kws={"alpha": 0.7, "color": "red"})
plt.title("Model Predictions")


# %%
resutls_none = pd.read_csv(data_dir + "none_fold_0_of_5_out.csv",
                           sep=";")
resutls_none.rename(columns={"y_true": "True Age", "y_pred": "Predicted Age"},
                    inplace=True)
fig, ax = plt.subplots(1, 1, figsize=[15, 10])

# sbn.scatterplot(data=resutls, x="y_true", y="y_pred", hue="site")
resutls.rename(columns={"y_true": "True Age", "y_pred": "Predicted Age"},
               inplace=True)
sbn.regplot(data=resutls_none, x="True Age", y="Predicted Age", ci=100,
            x_jitter=0, scatter_kws={"s": 10, "color": "blue"},
            ax=ax, line_kws={"alpha": 0.7, "color": "red"})
# %%
data_dir = "/home/nnieto/Nico/Harmonization/results_regression/test_regression_balanced_data_rvr_stack_rvr_pred/"
resutls = pd.read_csv(data_dir + "none_fold_0_of_5_out.csv",
                      sep=";")
resutls.rename(columns={"y_true": "True Age", "y_pred": "Predicted Age"},
               inplace=True)
fig, ax = plt.subplots(1, 1, figsize=[15, 10])
sbn.scatterplot(data=resutls, x="True Age", y="Predicted Age", hue="site",
                ax=ax)
sbn.regplot(data=resutls, x="True Age", y="Predicted Age", ci=100, x_jitter=0,
            scatter_kws={"s": 0}, ax=ax,
            line_kws={"alpha": 0.7, "color": "red"})
# %% Plot classification predictions

data_dir = "/home/nnieto/Nico/Harmonization/result_classification/test_classification_gender_all_bigs_logit_stack_gssvm_pred_5repetitions/"
resutls = pd.read_csv(data_dir + "pretend_fold_0_of_5_out.csv",
                      sep=";")

resutls.rename(columns={"y_true": "Gender", "y_pred": "Female probability",
                        "site": "Site ID"},
               inplace=True)
resutls["Gender"].replace({0: "Male", 1: "Female"},
                          inplace=True)

fig, ax = plt.subplots(1, 1, figsize=[10, 15])
sbn.swarmplot(data=resutls, y="Site ID", x="Female probability", hue="Gender",
              ax=ax)
# %%
