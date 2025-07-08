import numpy as np


def evaluate(cv_x):
    print(f"RMSE: {np.mean(cv_x['test_rmse'])}")
    print(f"RMSE: {np.mean(cv_x['test_mae'])}")
