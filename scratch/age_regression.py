# %%
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
from juharmonize import JuHarmonize, JuHarmonizeClassifier, JuHarmonizeRegressor # noqa
from juharmonize.utils import subset_data
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression # noqa
from sklearn.ensemble import RandomForestRegressor  # noqa
from skrvm import RVR
from sklearn.metrics import (                       # noqa
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV    # noqa


def get_JuHarmonizeModel(
    problem_type,
    pred_model,
    stack_model,
    n_splits=10,
    random_state=None,
    regression_points=10,
    regression_search=False,
    regression_search_tol=0,
    predict_ignore_site=False,
):
    if problem_type == "binary_classification":
        model = JuHarmonizeClassifier(
            pred_model=pred_model,
            n_splits=n_splits,
            stack_model=stack_model,
            use_cv_test_transforms=True,
            random_state=random_state,
            predict_ignore_site=predict_ignore_site,
        )
    else:
        model = JuHarmonizeRegressor(
            pred_model=pred_model,
            n_splits=n_splits,
            regression_points=regression_points,
            stack_model=stack_model,
            use_cv_test_transforms=True,
            random_state=random_state,
            regression_search=regression_search,
            regression_search_tol=regression_search_tol,
            predict_ignore_site=predict_ignore_site,
        )
    return model


def compute_metric(model, X, y, predicted_y):
    predicted_y_loop = model.predict(X)
    MAE = mean_absolute_error(y, predicted_y_loop)
    R2 = r2_score(y, predicted_y_loop)
    predicted_y.append(predicted_y_loop)
    return predicted_y, MAE, R2


def compute_metric_pretend(model, X, y, sites, predicted_y):
    predicted_y_loop = model.predict(X, sites=sites)
    MAE = mean_absolute_error(y, predicted_y_loop)
    R2 = r2_score(y, predicted_y_loop)
    predicted_y.append(predicted_y_loop)
    return predicted_y, MAE, R2


Y_1000brains = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_1000Gehirne_crop.csv")  # noqa
Y_Camcan = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_CamCAN_crop.csv")           # noqa
Y_ID1000 = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_ID1000.csv")                # noqa
Y_eNKI = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_eNKI_crop.csv")               # noqa

data = pd.concat([Y_1000brains, Y_Camcan, Y_eNKI, Y_ID1000])


# %%
X_1000brains = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_1000Gehirne_crop.csv", index_col=0)  # noqa
X_Camcan = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_CamCAN_crop.csv", index_col=0)           # noqa
X_ID1000 = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_ID1000.csv", index_col=0)                # noqa
X_eNKI = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_eNKI_crop.csv", index_col=0)               # noqa

X_total = pd.concat([X_1000brains, X_Camcan, X_eNKI, X_ID1000])
X_total.dropna(axis=1, inplace=True)
X = X_total.to_numpy()
# data["site"].replace({"ID1000": 0,
#                       "1000Gehirne": 1,
#                       "eNKI": 2,
#                       "CamCAN": 3},
#                      inplace=True)


# %%
# N_images = 138

# # Create a list of unique site names in the data dataframe
# sites = data['site'].unique()   # type: ignore

# # Create an empty dataframe to store the filtered data
# filtered_data = pd.DataFrame()
# filtered_data_X = pd.DataFrame()

# # For each site, randomly select N_images and append them
# # to the filtered_data dataframes
# for site in sites:
#     # Filter the data dataframe for the current site
#     site_data = data[data['site'] == site]

#     # Randomly select N_images from the current site's data dataframe
#     selected_indices = np.random.choice(len(site_data), N_images,
#                                         replace=False)
#     selected_data = site_data.iloc[selected_indices]

#     # Filter the data_X dataframe for the selected rows of data
#     selected_data_X = X_total.iloc[selected_data.index]

#     # Append the selected data and data_X to the filtered dataframes
#     filtered_data = filtered_data.append(selected_data)
#     filtered_data_X = filtered_data_X.append(selected_data_X)
# X_total = filtered_data_X
# data = filtered_data

fig = plt.figure(figsize=[20, 10])

ax = fig.add_subplot(1, 1, 1)
sbn.swarmplot(data=data, x="age", y="site", ax=ax, hue="gender")


X = X_total.to_numpy()

sites = data["site"].to_numpy()
Y = data["age"].to_numpy()
scaler = RobustScaler()
# models_pars = {"n_estimators": [5, 10, 25, 50, 100]}
pred_model = RVR(kernel="poly", degree=1)
model_none = RVR(kernel="poly", degree=1)
# pred_model = GridSearchCV(model, param_grid=models_pars, n_jobs=-1)
stack_model = RVR(kernel="poly", degree=1)

cheat_model = JuHarmonize()
# X = scaler.fit_transform(X, Y)
X_cheat = cheat_model.fit_transform(X, Y, sites)
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                             random_state=43)
# pretend
harm_model = get_JuHarmonizeModel(problem_type="regression",
                                  pred_model=pred_model,
                                  stack_model=stack_model,
                                  n_splits=10,
                                  random_state=None,
                                  regression_points=100,
                                  regression_search=False,
                                  regression_search_tol=0,
                                  predict_ignore_site=False,
                                  )

print(data["site"].value_counts())
target_model = JuHarmonize()

# %%
# %%

pred_cheat = []
pred_none = []

pred_cheat_trn = []
pred_none_trn = []


y_test_list = []
y_train_list = []
sites_train_list = []
sites_test_list = []
folds_lits = []

pred_pretend = []
pred_pretend_trn = []

pred_target = []
pred_target_trn = []
result = []

for i_fold, (train_index, test_index) in enumerate(kf.split(X, sites)):
    print("FOLD: " + str(i_fold))

    X_train, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X, sites, Y
    )

    X_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X, sites, Y)

    sites_train_list.append(sites_train)
    sites_test_list.append(sites_test)
    y_test_list.append(y_test)
    y_train_list.append(y_train)

    # none
    model_none.fit(X_train, y_train)
    pred_none, MAE, R2 = \
        compute_metric(model_none, X_test, y_test, pred_none)

    pred_none_trn, MAE_trn, R2_trn = \
        compute_metric(model_none, X_train, y_train, pred_none_trn)
    result.append([i_fold, "None", MAE, R2, MAE_trn, R2_trn])

    # target
    X_harm = target_model.fit_transform(X_train, y_train, sites_train)
    model_none.fit(X_harm, y_train)

    X_test_harm = target_model.transform(X_test, y_test, sites_test)
    pred_target, MAE, R2 = \
        compute_metric(model_none, X_test, y_test,
                       pred_target)

    pred_target_trn, MAE_trn, R2_trn = \
        compute_metric(model_none, X_train, y_train,
                       pred_target_trn)

    result.append([i_fold, "Target", MAE, R2, MAE_trn, R2_trn])

    # JuHarmonize
    harm_model.fit(X_train, y_train, sites_train)

    pred_pretend, MAE, R2 = compute_metric_pretend(
        harm_model, X_test, y_test, sites_test, pred_pretend)

    pred_pretend_trn, MAE_trn, R2_trn = \
        compute_metric_pretend(harm_model, X_train, y_train,
                               sites_train, pred_pretend_trn)
    result.append([i_fold, "JuHarmonize", MAE, R2, MAE_trn, R2_trn])

    # Cheat
    X_train_cheat, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X_cheat, sites, Y)

    X_test_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X_cheat, sites, Y)

    model_none.fit(X_train_cheat, y_train)
    pred_cheat, MAE, R2 = \
        compute_metric(model_none, X_test_test, y_test, pred_cheat)

    pred_cheat_trn, acc_cheat_trn, auc_cheat_trn = \
        compute_metric(model_none, X_train_cheat, y_train, pred_cheat_trn)
    result.append([i_fold, "Cheat", MAE, R2, MAE_trn, R2_trn])

# %%
result = pd.DataFrame(result, columns=["Fold", "Model", "MAE", "R2",
                                       "MAE Train", "R2 Train"])

# %%
# concatenate the arrays in pred_cheat into a single
# column called "predictions"
predictions = np.concatenate(pred_cheat)
df_cheat = pd.DataFrame({'predictions': predictions})

# concatenate the arrays in sites_test into a single column called "site"
site = np.concatenate(sites_test_list)
df_cheat['site'] = site

_y_true = np.concatenate(y_test_list)
df_cheat['true'] = _y_true
df_cheat['Harmonization scheme'] = "Cheat"

df_cheat["y_diff"] = df_cheat["true"] - df_cheat["predictions"]

predictions = np.concatenate(pred_none)
df_none = pd.DataFrame({'predictions': predictions})

# concatenate the arrays in sites_test into a single column called "site"
site = np.concatenate(sites_test_list)
df_none['site'] = site

_y_true = np.concatenate(y_test_list)
df_none['true'] = _y_true
df_none['Harmonization scheme'] = "None"

df_none["y_diff"] = df_none["true"] - df_none["predictions"]

predictions = np.concatenate(pred_pretend)
df_pred = pd.DataFrame({'predictions': predictions})
df_pred['site'] = site

_y_true = np.concatenate(y_test_list)
df_pred['true'] = _y_true
df_pred['Harmonization scheme'] = "JuHarmonize"

df_pred["y_diff"] = df_pred["true"] - df_pred["predictions"]

predictions = np.concatenate(pred_target)
df_target = pd.DataFrame({'predictions': predictions})
df_target['site'] = site

_y_true = np.concatenate(y_test_list)
df_target['true'] = _y_true
df_target['Harmonization scheme'] = "Target"

df_target["y_diff"] = df_target["true"] - df_target["predictions"]

df = pd.concat([df_pred, df_cheat, df_target, df_none])
# %%
# Plot
pal = sbn.cubehelix_palette(3, rot=-.5, light=0.5, dark=0.2)
_, ax = plt.subplots(1, 1, figsize=[20, 10])
sbn.boxenplot(data=df, x="site", y="y_diff", hue="Harmonization scheme",
              ax=ax, palette=pal,
              order=["ID1000", "eNKI", "CamCAN", "1000Gehirne"])
plt.grid(axis="y", color="black")
plt.ylabel("Age Difference [years]")
plt.xlabel("Site")

# %%
df.rename(columns={"true": "True Age", "predictions": "Predicted Age"},
          inplace=True)
fig, ax = plt.subplots(1, 1, figsize=[15, 10])

harm_mode = "JuHarmonize"
df_plot = df[df["Harmonization scheme"] == harm_mode]
sbn.scatterplot(data=df_plot, x="True Age", y="Predicted Age", hue="site",
                ax=ax)
sbn.regplot(data=df_plot, x="True Age", y="Predicted Age", ci=100, x_jitter=0,
            scatter_kws={"s": 0}, ax=ax,
            line_kws={"alpha": 0.7, "color": "red"})
plt.title("Harmonization Scheme: " + harm_mode)
# %%


sbn.boxplot(data=result, x="Model", y="MAE")
# %%

# %%
