# %%
# %%
import pandas as pd
import os
import sys
from prettyharmonize import PrettYharmonizeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from neuroHarmonize import harmonizationLearn, harmonizationApply

project_root = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)
from lib.data_processing import compute_classification_results          # noqa
from lib.data_loading import load_eICU              # noqa
# %%

# Directions
data_dir = "/data/eICU/"
save_dir = project_root+"/output/sepsis_classification_eicu/"

data = load_eICU(data_dir)
ABG_of_interes = ["paO2", "paCO2", "pH", "Base Excess",
                  "Hgb", "glucose", "bicarbonate", "lactate"]

min_images_per_site = 50
# Remove sites with less than a thd number of patients
site_counts = data["site"].value_counts()

# filter the site_ids with less than a thd
mask = site_counts[site_counts > min_images_per_site].index.tolist()

# Filter the sites with the minimun number of patietes
data = data[data['site'].isin(mask)]

# Separate the DataFrame into 'Alive' and 'Expired'
alive_df = data[data['endpoint'] == 'Alive']
expired_df = data[data['endpoint'] == 'Expired']

expired_final = pd.DataFrame()
alive_final = pd.DataFrame()
random_state = 23
sites_list = expired_df['site'].unique()

for site in sites_list:

    expired_site = expired_df[expired_df['site'] == site]
    alive_site = alive_df[alive_df['site'] == site]

    min_samples = min(len(expired_site), len(alive_df))

    print(min_samples)
    if min_samples == 0:
        print("no patients for one site")
        continue
    alive_final = pd.concat([alive_final,
                             alive_site.sample(n=min_samples,
                                               random_state=random_state)])
    expired_final = pd.concat([expired_final,
                               expired_site.sample(n=min_samples,
                                                   random_state=random_state)])

# Combine the remaining 'Alive' and 'Expired' patients
# into a new balanced DataFrame
balanced_df = pd.concat([alive_final, expired_final])

# Calculate the count of each target value (Alive, Expired) for each site
site_target_counts = balanced_df.groupby(['site', 'endpoint']).size().unstack(fill_value=0)     # noqa

# Calculate the proportion for each target within each site
site_target_proportions = site_target_counts.div(
    site_target_counts.sum(axis=1),
    axis=0)

# Combine counts and proportions into a single DataFrame
site_summary = pd.concat([site_target_counts, site_target_proportions],
                         axis=1, keys=['Count', 'Proportion'])

# Display the combined DataFrame
print(site_summary)
# %%
#
X = balanced_df.loc[:, ABG_of_interes].to_numpy()
sites = balanced_df["site"].reset_index()
Y = balanced_df["endpoint"].replace({"Expired": 1, "Alive": 0}).to_numpy()


# %%
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=1,
                                 random_state=23)


covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])


harm_cheat, data_cheat_no_target = harmonizationLearn(data=X, # noqa
                                                      covars=covars)

covars['Target'] = Y.ravel()

harm_cheat, data_cheat = harmonizationLearn(data=X, # noqa
                                            covars=covars)
data_cheat = pd.DataFrame(data_cheat)
clf = LogisticRegression()
PrettYharmonize_model = PrettYharmonizeClassifier(stack_model="logit",
                                                  pred_model="logit")

pred_PrettYharmonize = []
pred_cheat = []
pred_leakage = []
pred_none = []
pred_notarget = []
pred_y_true_loop = []


# %%
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train = X[train_index, :]
    X_cheat_train = data_cheat.iloc[train_index, :]
    X_cheat_no_target_train = data_cheat_no_target[train_index, :]

    site_train = sites.iloc[train_index, :]
    Y_train = Y[train_index]

    # Patients used to generete a prediction
    X_test = X[test_index, :]
    X_cheat_test = data_cheat.iloc[test_index, :]
    X_cheat_no_target_test = data_cheat_no_target[test_index, :]

    site_test = sites.iloc[test_index, :]

    Y_test = Y[test_index]
    pred_y_true_loop.append(Y_test)
    # Unharmonize model
    clf.fit(X_train, Y_train)
    pred_test = clf.predict_proba(X_test)[:, 1]
    results = compute_classification_results(i_fold, "Unharmonize Test", pred_test, Y_test, results)                 # noqa
    pred_none.append(pred_test)
    pred_train = clf.predict_proba(X_train)[:, 1]
    results = compute_classification_results(i_fold, "Unharmonize Train", pred_train, Y_train, results)                 # noqa

    # WHD
    clf.fit(X_cheat_train, Y_train)
    pred_test = clf.predict_proba(X_cheat_test)[:, 1]
    results = compute_classification_results(i_fold, "WDH Test", pred_test, Y_test, results)                 # noqa
    pred_cheat.append(pred_test)

    pred_train = clf.predict_proba(X_cheat_train)[:, 1]
    results = compute_classification_results(i_fold, "WDH Train", pred_train, Y_train, results)                 # noqa

    # # TTL
    covars_train = pd.DataFrame(site_train["site"].to_numpy(),
                                columns=['SITE'])
    covars_train['Target'] = Y_train.ravel()

    harm_model, harm_data = harmonizationLearn(X_train, covars_train)
    # Fit the model with the harmonizezd trian
    clf.fit(harm_data, Y_train)
    # covars
    covars_test = pd.DataFrame(site_test["site"].to_numpy(),
                               columns=['SITE'])
    covars_test['Target'] = Y_test.ravel()

    harm_data_test = harmonizationApply(X_test,
                                        covars_test,
                                        harm_model)

    pred_test = clf.predict_proba(harm_data_test)[:, 1]
    results = compute_classification_results(i_fold, "TTL Test", pred_test, Y_test, results)                 # noqa
    pred_leakage.append(pred_test)

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_classification_results(i_fold, "TTL Train", pred_train, Y_train, results)                 # noqa

    # No Target
    covars_train = pd.DataFrame(site_train["site"].to_numpy(),
                                columns=['SITE'])

    harm_model, harm_data = harmonizationLearn(X_train, covars_train)
    # Fit the model with the harmonizezd trian
    clf.fit(harm_data, Y_train)
    # covars
    covars_test = pd.DataFrame(site_test["site"].to_numpy(),
                               columns=['SITE'])
    harm_data_test = harmonizationApply(X_test,
                                        covars_test,
                                        harm_model)

    pred_test = clf.predict_proba(harm_data_test)[:, 1]
    results = compute_classification_results(i_fold, "No Target Test", pred_test, Y_test, results)                 # noqa
    pred_notarget.append(pred_test)

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_classification_results(i_fold, "No Target Train", pred_train, Y_train, results)                 # noqa

    # # PrettYharmonize
    PrettYharmonize_model.fit(X=X_train, y=Y_train,
                              sites=site_train["site"].to_numpy())
    pred_test = PrettYharmonize_model.predict_proba(X_test,
                                                sites=site_test["site"].to_numpy())[:, 1]           # noqa
    results = compute_classification_results(i_fold, "PrettYharmonize Test", pred_test, Y_test, results)                 # noqa
    pred_PrettYharmonize.append(pred_test)

    pred_train = PrettYharmonize_model.predict_proba(X_train,
                                                 sites=site_train["site"].to_numpy())[:, 1]         # noqa
    results = compute_classification_results(i_fold, "PrettYharmonize Train", pred_train, Y_train, results)                 # noqa


# %%

result_df = pd.DataFrame(results,
                         columns=["Fold",
                                  "Model",
                                  "Balanced ACC",
                                  "AUC",
                                  "F1",
                                  "Recall",
                                  ])

results.to_csv(save_dir+"eiCU_results_dependance.csv")

pd.DataFrame(pred_PrettYharmonize).to_csv(save_dir + "pred_PrettYharmonize.csv")            # noqa
pd.DataFrame(pred_cheat).to_csv(save_dir + "pred_wdh.csv")
pd.DataFrame(pred_leakage).to_csv(save_dir + "pred_ttl.csv")
pd.DataFrame(pred_none).to_csv(save_dir + "pred_unharmonize.csv")
pd.DataFrame(pred_notarget).to_csv(save_dir + "pred_notarget.csv")
pd.DataFrame(pred_y_true_loop).to_csv(save_dir + "y_true_loop.csv")
