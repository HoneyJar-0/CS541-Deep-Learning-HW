import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, norm
import scipy
import math
def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return A*B + np.transpose(C)

def problem_1c (x, y):
    return np.transpose(x).dot(y)

def problem_1d (A, j):
    return np.sum(A[::2,j])

def problem_1e (A, c, d):
    return np.mean(A[np.logical_and(A >= c, A <= d)])

def problem_1f (x, k, m, s):
    #x = column vector of length n
    #k = int
    #m,s= positive scalars
    #return n*k matrix
    z = np.ones(np.shape(x))
    I = np.identity(np.shape(x)[0])
    return np.random.multivariate_normal((x + m*z), s*I, k, tol=1).transpose()

def problem_1g (A):
    return np.apply_along_axis(np.random.permutation, 0, A)

def problem_1h (x):
    return ((x - np.mean(x))/np.std(x))

def problem_1i (x, k):
    return np.reshape(np.repeat(x,k),(np.shape(x)[0], k))


def linear_regression (X_tr, y_tr):
    #Eq from slides: (XX^T)^-1 Xy
    #Modified to fit X from starter code:
    #   (X^TX)^-1 X^Ty
    Xy = np.dot(X_tr.transpose(), y_tr)
    XXT = np.dot(X_tr.transpose(),X_tr)
    w = np.linalg.solve(XXT, Xy)
    return w

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("./age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("./age_regression_ytr.npy")
    X_te = np.reshape(np.load("./age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("./age_regression_yte.npy")

    w = linear_regression(X_tr, y_tr)

    # Report fMSE cost on the training and testing data (separately)
    yhat_tr = np.dot(X_tr, w)
    yhat_te = np.dot(X_te, w)

    #Calculate MSE
    fMSE_tr = np.mean((yhat_tr - y_tr)**2)
    fMSE_te = np.mean((yhat_te - y_te)**2)
    print(f"fMSE Training Data: {fMSE_tr}\nfMSE Testing Data: {fMSE_te}")

def part4a():
    poissonX = np.load("HW1\PoissonX.npy")
    plt.title("Empirical Probability Distribution")
    rates = [2.5,3.1,3.7,4.3]
    for rate in rates:
        poisson_dist = poisson.pmf(range(0, max(poissonX)+1), rate)
        plt.plot(range(0, max(poissonX)+1), poisson_dist, label=f'Poisson (λ={rate})')
    plt.hist(poissonX, density=True)
    plt.legend()
    plt.show()

print(1 - norm.cdf(0, loc=1, scale=(2 - 1/(1+ 1/math.e))**2))
train_age_regressor()
part4a()
