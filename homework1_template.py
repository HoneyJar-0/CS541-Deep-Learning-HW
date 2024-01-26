import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return ...

def problem_1c (x, y):
    return ...

def problem_1d (A, j):
    return ...

def problem_1e (A, c, d):
    return ...

def problem_1f (x, k, m, s):
    return ...

def problem_1g (A):
    return ...

def problem_1h (x):
    return ...

def problem_1i (x, k):
    return ...



def linear_regression (X_tr, y_tr):
    ...


def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
