# %%
import pandas as pd

pred = pd.read_csv("/home/nnieto/Nico/Harmonization/prediction_analysis/score_matrix.csv", index_col=0) # noqa
# %%
y_true = pd.read_csv("/home/nnieto/Nico/Harmonization/prediction_analysis/y_true.csv", index_col=0) # noqa

pred["mean"] = pred.mean(axis=1)
pred["std"] = pred.std(axis=1)
# %%
pred["True"] = y_true
# %%

all = pred["mean"] == pred["True"]
# %%
pred["std"].std()

# %%
