import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from typing import List, Union
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score


def compute_regression_results(i_fold: int, model: str,
                               pred: np.ndarray,
                               y: np.ndarray,
                               result: List[List[Union[int, str, float]]],
                               ) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.
    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.

        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """

    # Calculate the predictions using the passed ths

    # Compute all the metrics
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    age_bias = np.corrcoef(y, pred-y)[0, 1]

    # Append results
    result.append([i_fold, model, mae, r2, age_bias])

    return result


def compute_classification_results(i_fold: int, model: str,
                                   prob: np.ndarray,
                                   y: np.ndarray,
                                   result: List[List[Union[int, str, float]]],
                                   ) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.
    # noqa
    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # Calculate the predictions using the passed ths
    prediction = (prob > 0.5).astype(int)

    # Compute all the metrics
    bacc = balanced_accuracy_score(y, prediction)
    auc = roc_auc_score(y, prob)
    f1 = f1_score(y, prediction)
    recall = recall_score(y, prediction)

    # Append results
    result.append([i_fold, model,
                   bacc, auc, f1,
                   recall,
                   ])

    return result


def balance_gender(data, min_images):
    # min images 59 was set after analizing the number of images offline
    male = data[data["gender"] == "M"]
    female = data[data["gender"] == "F"]

    male = male.sample(n=min_images, random_state=23)
    female = female.sample(n=min_images, random_state=23)

    data_balanced = pd.concat([male, female])

    return data_balanced


def retain_images(X, Y):
    return X.loc[Y.index, :]


def get_min_common_number_images_in_age_bins(Y_data, age_bins):
    for t, n in enumerate(age_bins):
        if t == 0:
            age_low = n
            min_image = 1000
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age*idx_age2]
            # replace the age for the next bin
            age_low = age_high
            # Get the minimun value for each bin
            min_image_new = Y_filt["gender"].value_counts().min()
            if min_image_new < min_image:
                min_image = min_image_new

    return min_image


def filter_age_bins_with_qc(Y_data, age_bins, n_images, sampling):
    bin = 0
    filter_index = pd.Index([])  # Initialize an empty index

    for t, n in enumerate(age_bins):
        if t == 0:
            age_low = n
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age * idx_age2]

            if sampling == 'high_Q':
                # Sort by IQR in ascending order to get the lowest IQR values
                Y_filt = Y_filt.sort_values(by='IQR', ascending=True)
            elif sampling == 'low_Q':
                # Sort by IQR in descending order to get the highest IQR values
                Y_filt = Y_filt.sort_values(by='IQR', ascending=False)
            else:
                # Random sampling
                None

            # Sample n_images per gender
            males = Y_filt[Y_filt["gender"] == "M"].iloc[:n_images].index
            females = Y_filt[Y_filt["gender"] == "F"].iloc[:n_images].index

            if bin == 0:
                filter_index = males.append(females)
                bin = 1
            else:
                filter_index = filter_index.append(males)
                filter_index = filter_index.append(females)

            # Replace the age for the next bin
            age_low = age_high

    return filter_index
