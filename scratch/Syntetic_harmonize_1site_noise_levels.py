# %%
import pandas as pd
import numpy as np
from sklearn import svm         # noqa
import seaborn as sbn
from sklearn.linear_model import ElasticNet  # noqa
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold  # noqa
import matplotlib.pyplot as plt
from neuroHarmonize import harmonizationLearn, harmonizationApply
from skrvm import RVR       # noqa
from pathlib import Path
import sys
from sklearn.metrics import mean_absolute_error
dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from juharmonize.utils import subset_data # noqa
from lib.harmonize import eval_harmonizer, train_harmonizer  # noqa
# set numpy seed for reproducibility
np.random.seed(42)
# %%
# read data
data_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"

kf = KFold(n_splits=5)

features = 3747
# noise_level = [1, 1.05, 1.15, 1.2,
#                1.25, 1.3, 1.35, 1.4,
#                1.45, 1.5, 2, 5]

noise_level = [1.55, 1.6, 1.65, 1.7,
               1.75, 1.8, 1.85, 1.9,
               1.95, 2.5, 3, 4]

noise_level = [2.25, 2.75, 3.25, 3.75,
               4.25, 4.5, 4.75, 5.5,
               6, 6.5, 7, 8]
noise_level = [9, 10, 15]
noise_level = [17, 20, 25]

noise_level = [0.95, .90, .85, .80, .75, .70,
               0.65, .60, .55, .50, .45, 40,
               .35, .30, .25, .20, .15, .10, 0.05]

noise_level = [0.95, .90, .85, .80, .75, .70,
               0.65, .60, .55, .50, .45, .40,
               .35, .30, .25, .20, .15, .10, 0.05]

# noise_level = [1, 1.2]
# Load one site
x = pd.read_csv(data_dir+'X_eNKI.csv')
x = x.iloc[:, 0:features]
y = pd.read_csv(data_dir+'Y_eNKI.csv')['age']
site_initial = pd.read_csv(data_dir+'Y_eNKI.csv')['site']

# Generate site 1 form data
x1 = x.loc[0:408, :]
y1 = y.loc[0:408]
site1 = site_initial.loc[0:408]

# Generate site 2
x2 = x.loc[409::, :]
y2 = y.loc[409::]
site2 = site_initial.loc[409::]

# Put data together
y = pd.concat([y1, y2])
y = y.to_numpy()

model = svm.SVR(kernel="linear")
sites = np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))
covars = pd.DataFrame(sites.astype(int), columns=['SITE'])
covars['AGE'] = y
results_noise = []

print("Normal fit")
X = pd.concat([x1, x2])

pred = cross_val_predict(model, X, y, cv=kf)


results_original = pd.DataFrame({'y_pred': (pred)})
results_original["y_true"] = y
results_original["site"] = sites
results_original["Harmonization Mode"] = "Original"
results_original["noise_level"] = 1
results_original["MAE"] = mean_absolute_error(y, pred)

# results_mae = pd.DataFrame({'Mae': (mean_absolute_error(y, pred))
#                             "noise_level": 1,
#                             "Harmonization Mode": "Original"})

for noise in noise_level:
    # Induce site effect
    vars = x1.var(axis=0)
    means = x1.mean(axis=0)
    # gamma is feature-wise additive effect
    gamma = 0
    gamma = np.tile(gamma, (x1.shape[0], 1))
    # delta is feature-wise multiplicative effect
    delta = noise
    delta = np.tile(delta, (x1.shape[0], 1))
    # alpha is site-wise additive effect
    alpha = 0
    # uncombat the data
    x1site = delta*(x1 - alpha) + alpha + gamma

    # induce site effect
    vars = x2.var(axis=0)
    means = x2.mean(axis=0)
    # gamma is feature-wise additive effect
    gamma = 0
    gamma = np.tile(gamma, (x2.shape[0], 1))
    # delta is feature-wise multiplicative effect
    delta = 25
    delta = np.tile(delta, (x2.shape[0], 1))
    # alpha is site-wise additive effect
    alpha = 0
    # uncombat the data
    x2site = delta*(x2 - alpha) + alpha + gamma
    X_unharmonize = pd.concat([x1site, x2site]).to_numpy()

    print("Noise fit with noise: " + str(noise))
    predx = cross_val_predict(model, X_unharmonize, y, cv=kf)

    results_unharmonize = pd.DataFrame({'y_pred': (predx)})
    results_unharmonize["y_true"] = y
    results_unharmonize["site"] = sites
    results_unharmonize["Harmonization Mode"] = "Noise Data"
    results_unharmonize["noise_level"] = noise
    results_unharmonize["MAE"] = mean_absolute_error(y, predx)
    results_original["noise_level"] = noise

    covars = pd.DataFrame(sites.astype(int), columns=['SITE'])
    covars['AGE'] = y

    harm_model, harm_data = harmonizationLearn(X_unharmonize, covars)

    predh = cross_val_predict(model, harm_data, y, cv=kf)

    results_Cheat = pd.DataFrame({'y_pred': (predh)})
    results_Cheat["y_true"] = y
    results_Cheat["site"] = sites
    results_Cheat["Harmonization Mode"] = "Cheat"
    results_Cheat["noise_level"] = noise
    results_Cheat["MAE"] = mean_absolute_error(y, predh)

    # % JuHarmonize
    pred_model = svm.SVR(kernel="linear")
    stack_model = svm.SVR(kernel="linear")

    out_fold_list = []
    y_test_list = []
    sites_test_list = []
    covars = None
    regression_params = {
        "regression_points": 10,
        "regression_search_tol": 2.0,
        "regression_search": True,
    }
    kf = KFold(n_splits=5)
    sites = np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))
    for i_fold, (train_index, test_index) in enumerate(kf.split(X_unharmonize)):    # noqa
        print(i_fold)
        X_train, sites_train, y_train, covars_train, _ = subset_data(train_index, X_unharmonize, sites, y, covars)      # noqa
        X_test, sites_test, y_test, covars_test, _ = subset_data(test_index, X_unharmonize, sites, y, covars)           # noqa

        harm_model = train_harmonizer(
            "pretend",
            X_train,
            y_train,
            sites_train,
            covars_train,
            pred_model=pred_model,
            stack_model=stack_model,
            random_state=42,
            regression_params=regression_params,
            n_splits=5
        )

        out_fold, acc_fold = eval_harmonizer(
            harm_model, X_test, y_test, sites_test, covars_test
        )
        out_fold_list.append(out_fold)
        y_test_list.append(y_test)
        sites_test_list.append(sites_test)

    results_JuHarmonize = pd.DataFrame({'y_pred':
                                        np.concatenate(out_fold_list)})
    results_JuHarmonize["y_true"] = np.concatenate(y_test_list)
    results_JuHarmonize["site"] = np.concatenate(sites_test_list)
    results_JuHarmonize["Harmonization Mode"] = "JuHarmonize"
    results_JuHarmonize["noise_level"] = noise
    results_JuHarmonize["MAE"] = mean_absolute_error(np.concatenate(y_test_list),       # noqa
                                                     np.concatenate(out_fold_list))     # noqa

##############################################################################
    #
    stack_model = svm.SVR(kernel="linear")
    pred_leakage_list = []
    y_test_list = []
    sites_test_list = []
    harm_test_data = []
    sites = np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))

    for i_fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(i_fold)
        covars = None
        X_train, sites_train, y_train, covars_train, _ = subset_data(train_index, X_unharmonize, sites, y, covars)      # noqa
        X_test, sites_test, y_test, covars_test, _ = subset_data(test_index, X_unharmonize, sites, y, covars)           # noqa
        
        covars_train = pd.DataFrame(sites_train.astype(int), columns=['SITE'])
        covars_train['AGE'] = y_train
        # Fit and transorm train data
        harm_model, harm_data = harmonizationLearn(X_train, covars_train)
        # Fit the model with the harmonizezd trian
        model.fit(harm_data, y_train)

        # Testing
        covars_test = pd.DataFrame(sites_test.astype(int), columns=['SITE'])
        covars_test['AGE'] = y_test
        # Transform the test data
        harm_data = harmonizationApply(X_test, covars_test, harm_model)
        # generate a prediction
        harm_test_data.append(harm_data)
        pred_leakage = model.predict(harm_data)
        pred_leakage_list.append(pred_leakage)
        y_test_list.append(y_test)
        sites_test_list.append(sites_test)

    results_leakage = pd.DataFrame({'y_pred':
                                    np.concatenate(pred_leakage_list)})
    results_leakage["y_true"] = np.concatenate(y_test_list)
    results_leakage["site"] = np.concatenate(sites_test_list)
    results_leakage["Harmonization Mode"] = "Leakage"
    results_leakage["noise_level"] = noise
    results_leakage["MAE"] = mean_absolute_error(np.concatenate(y_test_list),
                                                 np.concatenate(pred_leakage_list)) # noqa

    results_noise.append(pd.concat([results_original,
                                    results_unharmonize, results_Cheat,
                                    results_JuHarmonize, results_leakage]))


results_noise_df = pd.DataFrame(pd.concat(results_noise))


sbn.barplot(data=results_noise_df,
            x="noise_level", y="MAE", hue="Harmonization Mode")

results_noise_df.to_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya_below_1.csv")  # noqa

# print("MAE Original: " + str(round(np.mean(np.abs(results_original["y_pred"] - results_original["y_true"])),3)))             # noqa
# print("MAE UnHarmonize: " + str(round(np.mean(np.abs(results_unharmonize["y_pred"] - results_unharmonize["y_true"])),3)))    # noqa

# print("MAE Cheat: " + str(round(np.mean(np.abs(results_Cheat["y_pred"] - results_Cheat["y_true"])),3)))                      # noqa

# print("MAE Leak: " + str(round(np.mean(np.abs(results_leakage["y_pred"] - results_leakage["y_true"])),3)))                   # noqa
# print("MAE JuHarmonize: " + str(round(np.mean(np.abs(results_JuHarmonize["y_pred"] - results_JuHarmonize["y_true"])),3)))    # noqa

# %%

plt.figure()
sbn.barplot(data=results_noise_df,
            x="noise_level", y="MAE", hue="Harmonization Mode")
plt.ylim(5, 16)
# %%
plt.figure()
sbn.lineplot(data=results_noise_df,
             x="noise_level", y="MAE", hue="Harmonization Mode")
plt.xlim(0, 1)
# %%
# %%

d1 = pd.read_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya.csv")  # noqa
d2 = pd.read_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya_last.csv")  # noqa
d3 = pd.read_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya_last2.csv")  # noqa

df = pd.concat([d1, d2, d3])
# %%
plt.figure()
sbn.lineplot(data=df,
             x="noise_level", y="MAE", hue="Harmonization Mode")
# plt.ylim(3, 30)
# %%

df_plot = df[df["Harmonization Mode"] == "Noise Data"]
plt.figure()
sbn.lineplot(data=df_plot,
             x="noise_level", y="MAE", hue="site")
# %%
plt.figure()
sbn.lineplot(data=df,
             x="noise_level", y="MAE", hue="Harmonization Mode")
plt.ylim(6, 7.5)

plt.xlim(0, 1.)

# %%

plt.figure()
sbn.lineplot(data=df,
             x="noise_level", y="MAE", hue="Harmonization Mode")
plt.ylim(5, 15)

plt.xlim(1, 3)
# %%
titles = "JuHarmonize"
noise_lvl = 2
df_plot = df[df["Harmonization Mode"] == titles]
df_plot = df_plot[df_plot["noise_level"] == noise_lvl]
fig, axes = plt.subplots(nrows=1, ncols=1)

sbn.scatterplot(data=df_plot, x="y_true", y="y_pred", hue="site")
mae = np.mean(np.abs(df_plot["y_pred"] - df_plot["y_true"]))
axes.set_title(titles + " with noise " + str(noise_lvl) + " MAE: " + r"$\bf{" + str(round(mae, 4)) + "}$")
# %%

harm_mode = "No Target"
df_plot = df[df["Harmonization Mode"] == harm_mode]

Mae_d0 = []
Mae_d1 = []

for noise_lvl in np.unique(df_plot["noise_level"]):

    df_plot_n = df_plot[df_plot["noise_level"]==noise_lvl]
    d0 = df_plot_n[df_plot_n["site"]==0]
    d1 = df_plot_n[df_plot_n["site"]==1]

    Mae_d0.append([noise_lvl, mean_absolute_error(d0["y_true"],d0["y_pred"]), 0])
    Mae_d1.append([noise_lvl, mean_absolute_error(d1["y_true"],d1["y_pred"]), 1])

df_MAE_plot0 = pd.DataFrame((Mae_d0))
df_MAE_plot1 = pd.DataFrame((Mae_d1))

df_MAE_plot = pd.concat([df_MAE_plot0, df_MAE_plot1])

df_MAE_plot.rename(columns={0:"noise_lvl",1:"MAE",2:"site"}, inplace=True)

plt.figure()
sbn.lineplot(data=df_MAE_plot,
             x="noise_lvl", y="MAE", hue="site")
plt.title(harm_mode)
# %%


harm_mode = "No Target"
df_plot = df[df["Harmonization Mode"] == harm_mode]

Mae_d0 = []
Mae_d1 = []
mad_mean = []

for noise_lvl in np.unique(df_plot["noise_level"]):

    df_plot_n = df_plot[df_plot["noise_level"]==noise_lvl]
    d0 = df_plot_n[df_plot_n["site"]==0]
    d1 = df_plot_n[df_plot_n["site"]==1]

    Mae_d0.append([noise_lvl, mean_absolute_error(d0["y_true"],d0["y_pred"]), 0])
    Mae_d1.append([noise_lvl, mean_absolute_error(d1["y_true"],d1["y_pred"]), 1])
    mad_mean.append([noise_lvl, mean_absolute_error(d1["y_true"],d1["y_true"].mean()), 0])
df_MAE_plot0 = pd.DataFrame((Mae_d0))
df_MAE_plot1 = pd.DataFrame((Mae_d1))

df_MAE_plot = pd.concat([df_MAE_plot0, df_MAE_plot1])

df_MAE_plot.rename(columns={0:"noise_lvl",1:"MAE",2:"site"}, inplace=True)

plt.figure()
sbn.lineplot(data=df_MAE_plot,
             x="noise_lvl", y="MAE", hue="site")
plt.title(harm_mode)


# %%
# mean models

mod_mean_d0 = []
mod_mean_d1 = []
mod_mean_all = []

for noise_lvl in np.unique(df_plot["noise_level"]):

    df_plot_n = df_plot[df_plot["noise_level"]==noise_lvl]
    d0 = df_plot_n[df_plot_n["site"]==0]
    d1 = df_plot_n[df_plot_n["site"]==1]

    mod_mean_d1.append([noise_lvl,
                        mean_absolute_error(d1["y_true"],d1["y_true"].mean()*np.ones(d1["y_true"].shape)), 1])
    mod_mean_d0.append([noise_lvl,
                        mean_absolute_error(d0["y_true"],d0["y_true"].mean()*np.ones(d0["y_true"].shape)), 0])
    mod_mean_all.append([noise_lvl,
                        mean_absolute_error(df_plot_n["y_true"],df_plot_n["y_true"].mean()*np.ones(df_plot_n["y_true"].shape)), 2])



df_MAE_plot0 = pd.DataFrame((mod_mean_d1))
df_MAE_plot1 = pd.DataFrame((mod_mean_d0))
df_MAE_plot = pd.DataFrame((mod_mean_all))

df_MAE_plot = pd.concat([df_MAE_plot0, df_MAE_plot1,df_MAE_plot])

df_MAE_plot.rename(columns={0:"noise_lvl",1:"MAE",2:"site"}, inplace=True)

plt.figure()
sbn.lineplot(data=df_MAE_plot,
             x="noise_lvl", y="MAE", hue="site")
plt.title("Mean")

# %% No Target


noise_level = [1, 1.05, 1.15, 1.2,
               1.25, 1.3, 1.35, 1.4,
               1.45, 1.5, 2, 5, 1.55, 1.6, 1.65, 1.7,
               1.75, 1.8, 1.85, 1.9,
               1.95, 2.5, 3, 4, 2.25, 2.75, 3.25, 3.75,
               4.25, 4.5, 4.75, 5.5, 6, 6.5, 7, 8,  9, 10, 15,
               17, 20, 25]

results_noise = []
for noise in noise_level:
    # Induce site effect
    vars = x1.var(axis=0)
    means = x1.mean(axis=0)
    # gamma is feature-wise additive effect
    gamma = 0
    gamma = np.tile(gamma, (x1.shape[0], 1))
    # delta is feature-wise multiplicative effect
    delta = noise
    delta = np.tile(delta, (x1.shape[0], 1))
    # alpha is site-wise additive effect
    alpha = 0
    # uncombat the data
    x1site = delta*(x1 - alpha) + alpha + gamma

    # induce site effect
    vars = x2.var(axis=0)
    means = x2.mean(axis=0)
    # gamma is feature-wise additive effect
    gamma = 0
    gamma = np.tile(gamma, (x2.shape[0], 1))
    # delta is feature-wise multiplicative effect
    delta = 25
    delta = np.tile(delta, (x2.shape[0], 1))
    # alpha is site-wise additive effect
    alpha = 0
    # uncombat the data
    x2site = delta*(x2 - alpha) + alpha + gamma
    X_unharmonize = pd.concat([x1site, x2site]).to_numpy()

    covars = pd.DataFrame(sites.astype(int), columns=['SITE'])

    harm_model, harm_data = harmonizationLearn(X_unharmonize, covars)

    predh = cross_val_predict(model, harm_data, y, cv=kf)
    print("noise lvl:" + str(noise))
    results_notarget = pd.DataFrame({'y_pred': (predh)})
    results_notarget["y_true"] = y
    results_notarget["site"] = sites
    results_notarget["Harmonization Mode"] = "No Target"
    results_notarget["noise_level"] = noise
    results_notarget["MAE"] = mean_absolute_error(y, predh)
    results_noise.append(pd.concat([results_notarget]))


results_noise_df = pd.DataFrame(pd.concat(results_noise))

# %%
results_noise_df.to_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya_notarget.csv")  # noqa

# %%

d1 = pd.read_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya.csv")  # noqa
d2 = pd.read_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya_last.csv")  # noqa
d3 = pd.read_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya_last2.csv")  # noqa
d4 = pd.read_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/results_noisydaya_notarget.csv")  # noqa

df = pd.concat([d1, d2, d3, d4])
# %%

# %%
