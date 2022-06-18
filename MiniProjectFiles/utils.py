import numpy as np
import pandas as pd
from datetime import datetime


def score(y_pred, y_true):
    """Computes RMSE over all three sites.

    Args:
        y_pred (numpy array): NxD array of predictions
        y_true (numpy array): NxD array of actual values

    Returns:
        float: rmse value
    """
    return np.sqrt(np.mean(np.square(y_true-y_pred)))


def make_submission_file(predictions, filename=None, path="submissions/"):
    """Creates calid submission file frmom dataframe or array.

    Args:
        predictions (pandas df or numpy array): Predicted values, muste be of shape 2904x3
        filename (str, optional): filename. Defaults to None.
        path (str, optional): file path. Defaults to "submissions/".

    Raises:
        ValueError: If predictions are not numpy arrays or pandas dataframe
        ValueError: If shape is not of dimension ()
    """
    
    if filename is None:
        filename = datetime.now().strftime("%Y%m%d%H%M%S")+"_my_submission.csv"
    
    # convert to numpy array
    if isinstance(predictions, pd.DataFrame):
        vals = predictions.values
    elif isinstance(predictions, np.ndarray):
        vals = predictions
    else:
        raise ValueError("\npredictions has to be either pandas DataFrame or numpy array.")
    
    # check for dimensions
    if vals.shape != (2904,3):
        raise ValueError(f"\npredictions is expected to have shape (2904,3) but has shape {predictions.shape}.")

    # check for NAs
    if np.sum(np.isnan(vals))>0:
        print(f"\npredictions contains {np.sum(np.isnan(vals))} NA values!")

    # check for negative values
    if np.sum(vals<0):
        print(f"\npredictions contains {np.sum(vals<0)} negative values.")    
    
    # check for values larger than nominal power
    if np.sum(vals>1.0):
        print(f"\npredictions contains {np.sum(vals>1.0)} values larger than 1.")

    # create pandas df and save to csv
    index = pd.Index(pd.date_range(start='2013/9/1 0:00', end='2013/12/30 23:00', freq='H'), name="datetime")
    columns=["energy_produced_1","energy_produced_2","energy_produced_3"]
    y_pred_df = pd.DataFrame(data=vals, index=index, columns=columns)
    y_pred_df.to_csv(path+filename, float_format="%.6f")
    print("\nSubmission .csv file created successfully!\n")
    