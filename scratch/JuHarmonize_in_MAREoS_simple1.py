# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from juharmonize import JuHarmonizeClassifier
from sklearn.metrics import balanced_accuracy_score
root_dir = "/home/nnieto/Nico/Harmonization/data/MAREoS/public_datasets/"
example = "eos_simple1"
data = pd.read_csv(root_dir+example+"_data.csv", index_col=0)
target = pd.read_csv(root_dir+example+"_response.csv", index_col=0)

sites = data["site"]
folds = data["folds"]
clf = LogisticRegression()
harm_model = JuHarmonizeClassifier(stack_model="svm", pred_model="rf")

print("Number of sites: " + str(sites.nunique()))
print("Number of classes: " + str(target.value_counts()))

harm_results = []
simple_results = []
for fold in folds.unique():
    print("Fold Number: " + str(fold))
    X = data[data["folds"] != fold]
    site_train = X["site"]
    X = X.drop(columns=["cov1", "cov2", "site", "folds"])
    y = target[data["folds"] != fold]

    X_test = data[data["folds"] == fold]
    site_test = X_test["site"]
    X_test = X_test.drop(columns=["cov1", "cov2", "site", "folds"])
    y_test = target[data["folds"] == fold]
    clf.fit(X, y)

    simple_results.append(balanced_accuracy_score(y_true=y_test, y_pred=clf.predict(X=X_test)))    # noqa

    harm_model.fit(X.to_numpy(), y.to_numpy(), sites=site_train.to_numpy())
    y_pred = harm_model.predict(X_test.to_numpy(), sites=site_test)
    harm_results.append(balanced_accuracy_score(y_true=y_test, y_pred=y_pred))

print("JuHarmonize bACC: " + str(np.array(harm_results).mean()))     # noqa
print("Logit clf bACC: " + str(np.array(simple_results).mean()))

# %%
