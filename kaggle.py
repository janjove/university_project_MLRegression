import pandas as pd
import numpy as np


def init_doc():
    X_kaggle = pd.read_csv('test_advance.csv')
    X_kaggle = fix_problem(X_kaggle)
    return X_kaggle
def fix_problem(X_kaggle):
    X_kaggle = X_kaggle.loc[X_kaggle['Id'] == 2121, 'TotalBsmtSF'] = 0
    return X_kaggle

def final_files(indices_car,indices_bar,y_pred_car, y_pred_bar):
    series_car = pd.Series(indices_car)
    series_bar = pd.Series(indices_bar)

    # Concatenate the Series
    all_indices = pd.concat([series_car, series_bar])

    # Create a new DataFrame using these indices
    submission = pd.DataFrame(all_indices, columns=['Id'])

    # Reset the index to have a clean DataFrame
    # Concatenate the two DataFrames to form the final submission DataFrame
    submission['SalePrice'] = np.concatenate([y_pred_car, y_pred_bar])

    # Reset the index for the final DataFrame
    submission.reset_index(drop=True, inplace=True)
    submission_file_path = '/APC_GIT/Data/submission.csv'

    submission.to_csv(submission_file_path,index=None)