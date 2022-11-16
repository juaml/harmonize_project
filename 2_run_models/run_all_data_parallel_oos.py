# %%:
import pandas as pd

import argparse
from pathlib import Path
import sys

to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)
from lib.harmonize import eval_harmonizer, train_harmonizer  # noqa
from lib import io  # noqa
from lib import ml  # noqa
from lib.logging import logger, configure_logging  # noqa
from lib.utils import check_params
# Running Parameters

configure_logging()

# parse parameters
parser = argparse.ArgumentParser(description="Attributes")

parser = io.set_argparse_params(parser, use_oos=True)
parser = ml.set_argparse_params(parser)

parser.add_argument(
    "--save_dir", type=str, default="", help="Save results path"
)

parser.add_argument(
    "--problem_type",
    type=str,
    default="binary_classification",
    help="Problem type",
)

parser.add_argument(
    "--harm_n_splits", type=int, default=10, help="Folds in Harmonization"
)
parser.add_argument(
    "--regression_points", type=int, default=100, help="Regression point"
)
parser.add_argument(
    "--regression_search", action='store_true', default=False,
    help="Do regression search"
)

parser.add_argument(
    "--regression_search_tol",
    type=float,
    default=2.0,
    help="Regression tolerance",
)

parser.add_argument(
    "--cutoff_age",
    type=int,
    default=-1,
    help=(
        "Limit age to binarize "
        "If -1 (default), not used"
    ),
)


parser.add_argument(
    "--harmonize_mode", type=str, default="JUHA", help="Harmonization Mode"
)
parser.add_argument(
    "--random_state", type=int, default=23, help="Random State use"
)

parser.add_argument(
    "--use_disk",
    action="store_true",
    default=False,
    help="Use disk for save RF",
)

parser.add_argument(
    "--n_jobs", type=int, default=None, help="Numbers of jobs for predictor"
)


params = parser.parse_args()

check_params(params)

# Paths
save_dir = Path(params.save_dir)

save_dir.mkdir(exist_ok=True, parents=True)

logger.info(f"Saving results in {save_dir.as_posix()}")

# General
random_state = params.random_state
problem_type = params.problem_type


# Harmonizaton set up
harm_n_splits = params.harm_n_splits
harmonize_mode = params.harmonize_mode
use_disk = params.use_disk
n_jobs = params.n_jobs


# ######################## Models Set ups

pred_model, stack_model = ml.get_models(params, problem_type)


regression_params = None
if problem_type == "regression":
    regression_params = {
        "regression_points": params.regression_points,
        "regression_search_tol": params.regression_search_tol,
        "regression_search": params.regression_search,
    }


# ######################## Data loading and preprocessing
data = io.get_MRI_data(params, problem_type, use_oos=True)
X, y, sites, covars, Xoos, yoos, sitesoos, covarsoos = data  # type: ignore


harm_model = train_harmonizer(
    harmonize_mode,
    X,
    y,
    sites,
    covars,
    pred_model,
    stack_model=stack_model,
    random_state=random_state,
    n_splits=10,
    regression_params=regression_params,
    use_disk = use_disk,
    n_jobs=n_jobs
)

out_fold, acc_fold = eval_harmonizer(
    harm_model, Xoos, yoos, sitesoos, covarsoos
)


logger.info("================================")
logger.info(f"\tOOS - SCORE: {acc_fold}")
logger.info("================================")

out_fname = f"{harmonize_mode}_oos_out.csv"
to_save = pd.DataFrame({"y_true": yoos, "y_pred": out_fold, "site": sitesoos})
to_save['harmonize_mode'] = harmonize_mode
out_path = save_dir / out_fname
logger.info(f"Saving dataframe in {out_path.as_posix()}")
to_save.to_csv(out_path, sep=';')
