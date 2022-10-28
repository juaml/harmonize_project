# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor as RFR

from juharmonize import JuHarmonize
from juharmonize.utils import subset_data


import argparse
from pathlib import Path
import sys
import joblib
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)
from lib.harmonize import eval_harmonizer, train_harmonizer  # noqa
from lib import io  # noqa
from lib import ml  # noqa
from lib.logging import logger, configure_logging  # noqa
from lib.utils import check_params


features = 3610

random_vector = np.random.rand(1,features)
random_vector_1 = np.random.rand(1,features)

rf = RFR()

rf.fit(random_vector,random_vector_1)
 

joblib.dump(rf,"../rf")
# %%
