import numpy as np

from sklearn.metrics import (
    accuracy_score,
    r2_score,
)

from juharmonize import (
    JuHarmonize,
    JuHarmonizeRegressor,
    JuHarmonizeClassifier,
    JuHarmonizePredictor,
    JuHarmonizePredictorCV,
)

from .logging import logger


def get_JuHarmonizeModel(problem_type, pred_model, stack_model,
                         n_splits=10, random_state=None, 
                         regression_points=10,
                         regression_search=False,
                         regression_search_tol=0,
                         predict_ignore_site=False):
    if problem_type == "binary_classification":
        logger.info("Using JuHarmonizeClassifier")
        logger.info(f"\tpred_model: {pred_model}")
        logger.info(f"\tstack_model: {stack_model}")
        logger.info(f"\tn_splits: {n_splits}")
        logger.info(f"\trandom_state: {random_state}")
        logger.info(f"\tpredict_ignore_site: {predict_ignore_site}")
        logger.info("\tuse_cv_test_transforms: True")
        model = JuHarmonizeClassifier(
            pred_model=pred_model,
            n_splits=n_splits,
            stack_model=stack_model,
            use_cv_test_transforms=True,
            random_state=random_state,
            predict_ignore_site=predict_ignore_site)
    else:
        logger.info("Using JuHarmonizeRegressor")
        logger.info(f"\tpred_model: {pred_model}")
        logger.info(f"\tstack_model: {stack_model}")
        logger.info(f"\tn_splits: {n_splits}")
        logger.info(f"\tregression_points: {regression_points}")
        logger.info(f"\tregression_search: {regression_search}")
        logger.info(f"\tregression_search_tol: {regression_search_tol}")
        logger.info(f"\tpredict_ignore_site: {predict_ignore_site}")
        logger.info("\tuse_cv_test_transforms: True")
        logger.info(f"\trandom_state: {random_state}")
        model = JuHarmonizeRegressor(
            pred_model=pred_model,
            n_splits=n_splits,
            regression_points=regression_points,
            stack_model=stack_model,
            use_cv_test_transforms=True,
            random_state=random_state,
            regression_search=regression_search,
            regression_search_tol=regression_search_tol,
            predict_ignore_site=predict_ignore_site)
    return model


def train_harmonizer(
        harm_type, X, y, sites, covars, pred_model,
        stack_model=None, random_state=None, n_splits=10,
        regression_params=None):
    if regression_params is None:
        regression_params = {}
    problem_type = "regression"
    if len(np.unique(y)) == 2:
        problem_type = "binary_classification"

    logger.info(f"Training harmonizer {harm_type}")
    out = {}
    out['problem_type'] = problem_type
    out['harm_type']  = harm_type
    out['pred_model'] = pred_model
    out['harm_model'] = None
    if harm_type == 'none':
        logger.info("Predictive model fit")
        out['pred_model'].fit(X, y)
        logger.info("Fit done")
    elif harm_type in ['target']:
        # harmonize train and then apply to test but using labels
        out['harm_model'] = JuHarmonize()
        logger.info("JuHarmonize fit_transform")
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        logger.info("Predictive model fit")
        out['pred_model'].fit(X_harm, y)
        logger.info("Fit done")
    elif harm_type == 'notarget':
        out['harm_model'] = JuHarmonize(preserve_target=False)
        logger.info("JuHarmonize fit_transform")
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        logger.info("Predictive model fit")
        out['pred_model'].fit(X_harm, y)
        logger.info("Fit done")
    elif harm_type == 'predict':
        out['harm_model'] = JuHarmonizePredictor()
        logger.info("JuHarmonizePredictor fit_transform")
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        logger.info("Predictive model fit")
        out['pred_model'].fit(X_harm, y)
        logger.info("Fit done")
    elif harm_type in ['pretend', 'pretend_nosite']:
        assert stack_model is not None
        predict_ignore_site = harm_type == 'pretend_nosite'
        out['pred_model'] = get_JuHarmonizeModel(
            problem_type, pred_model, stack_model,
            predict_ignore_site=predict_ignore_site,
            random_state=random_state, n_splits=n_splits,
            **regression_params)
        logger.info("JuHarmonizeClassifier/Regressor fit")
        out['pred_model'].fit(X, y, sites, covars)
        logger.info("Fit done")
    elif harm_type in ['predict_pretend', 'predict_pretend_nosite']:
        assert stack_model is not None
        predict_ignore_site = harm_type == 'pretend_nosite'
        out['pred_model'] = get_JuHarmonizeModel(
            problem_type, pred_model, stack_model,
            predict_ignore_site=predict_ignore_site,
            random_state=random_state, n_splits=n_splits,
            **regression_params)
        out['harm_model'] = JuHarmonizePredictorCV()
        logger.info("JuHarmonizePredictorCV fit_transform")
        X_harm = out['harm_model'].fit_transform(X, y, sites, covars)
        logger.info("JuHarmonizeClassifier/Regressor fit")
        out['pred_model'].fit(X_harm, y, sites, covars)
        logger.info("Fit done")
    else:
        raise ValueError(f"Unknown harm_type {harm_type}")
    logger.info("Training done")
    return out


def eval_harmonizer(harm, X, y, sites, covars=None):
    logger.info(f"Evaluation harmonizer")
    if harm['harm_type'] == 'none':
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] == 'target':
        assert y is not None
        X = harm['harm_model'].transform(X, y, sites, covars)
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] == 'notarget':
        X = harm['harm_model'].transform(X, y, sites, covars)
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] == 'predict':
        X = harm['harm_model'].transform(X)
        out = harm['pred_model'].predict(X)
    elif harm['harm_type'] in ['pretend', 'pretend_nosite']:
        out = harm['pred_model'].predict(X, sites, covars)
    elif harm['harm_type'] in ['predict_pretend', 'predict_pretend_nosite']:
        X = harm['harm_model'].transform(X)
        out = harm['pred_model'].predict(X, sites, covars)
    else:
        raise ValueError(f"Unknown harm_type: {harm['harm_type']}")

    if harm['problem_type'] == 'binary_classification':
        acc = accuracy_score(y, out)
    else:
        acc = r2_score(y, out)
    logger.info(f"Evaluation done: {acc}")
    return out, acc
