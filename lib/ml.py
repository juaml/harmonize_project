from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
)
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from skrvm import RVR, RVC
from sklearn_rvm import EMRVC
from .utils import logger

_valid_models = {
    "binary_classification": ["gssvm", "rvc", "svm"],
    "regression": ["gsgpr", "gssvm", "ridgecv", "rvr"],
}


def set_argparse_params(parser):

    parser.add_argument(
        "--pca", action="store_true", default=False, help="Use PCA"
    )
    parser.add_argument(
        "--scaler",
        action="store_true",
        default=False,
        help="Use scaler (Robust if --pca is false, otherwise Standard)",
    )

    parser.add_argument(
        "--pred_model", type=str, default="svm", help="Prediction model to use"
    )
    parser.add_argument(
        "--stack_model", type=str, default="rf", help="Stacked model to use"
    )

    return parser


def get_models(params, problem_type):
    pred_model = params.pred_model
    stack_model = params.stack_model
    logger.info("Setting up models")
    logger.info(f"\tPrediction model: {pred_model}")
    logger.info(f"\tStack model: {stack_model}")
    pca = params.pca
    scaler = params.scaler
    logger.info(f"\tPCA: {pca}")
    logger.info(f"\tScaler: {scaler}")
    logger.info(f"\tUse Disk: {params.use_disk}")
    logger.info(f"\tN Jobs: {params.n_jobs}")
    if pred_model not in _valid_models[problem_type]:
        raise ValueError(
            f"Invalid prediction model ({pred_model}). "
            f"Must be one of {_valid_models[problem_type]}")
    # #### Clasiffiers parameters
    if problem_type == "binary_classification":
        if pred_model == "gssvm":
            params_svm = {
                "kernel": ("linear", "rbf"),
                "C": [0.01, 0.1, 1, 10, 100],
            }
            pred_model = GridSearchCV(SVC(probability=True), params_svm)
        elif pred_model == "svm":
            pred_model = SVC(probability=True, kernel="linear")
        elif pred_model == "rvc":
            pred_model = RVC(kernel="poly", degree=1)
    else:
        # ['gsgpr', 'gssvm', 'ridgecv', 'rvr'],
        if pred_model == "gssvm":
            params_svm = {
                "kernel": ("linear", "rbf"),
                "C": [0.01, 0.1, 1, 10, 100],
            }
            pred_model = GridSearchCV(SVR(), params_svm)

        elif pred_model == "gsgpr":
            kernel1 = RBF() + WhiteKernel()
            kernel2 = DotProduct() + WhiteKernel()
            params_gpr = {"kernel": [kernel1, kernel2]}
            params_gpr = [
                {
                    "alpha": [1e-5],
                    "kernel": [RBF(x, (1e-7, 10e7)) for x in [0.1, 1, 10]],
                }
            ]
            pred_model = GridSearchCV(
                GPR(n_restarts_optimizer=5, normalize_y=True), params_gpr
            )
        elif pred_model == "ridgecv":
            pred_model = RidgeCV()
        elif pred_model == "rvr":
            # pred_model = RVR(kernel="poly", degree=1)
            pred_model = EMRVC(kernel="poly", degree=1, gamma='scale', bias_used=True)   
        else:
            raise ValueError("Regression model not supported")

    if pca:
        pca80 = PCA(n_components=0.8, svd_solver="full")
        pred_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", pca80),
                ("model", pred_model),
            ]
        )
    elif scaler:
        pred_model = Pipeline(
            [("scaler", RobustScaler()), ("model", pred_model)]
        )

    return pred_model, stack_model
