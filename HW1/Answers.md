# HW 1 - JReiss

## Part 1 Code
```
import numpy as np

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


```
## Part 2
### 2a
The code is not subtracting the minimum row element from the respective row because the np.subtract() method is broadcasting row_min row-wise, causing [0,3,6] to repeat as a row rather than a column.
| Desired Behavior    | Current Behavior |
|:---------------------:|:------------------:|
| [0] →<br>[3] →<br>[6] → | [0 3 6]<br> ↓ ↓ ↓  |

### 2b
```
def problem2b(Y):
    #Assumptions:
    # Y is (3,3,3)
    # Row refers to the same values as in the code in 2a
    #   If this is not the case, axis would need to be changed:
    #   Shape = (depth, column, row) when mapped from X
    #     axis = 0: Rows are defined along depth
    #     axis = 1: Rows are defined as columns in each layer of depth
    #     axis = 2: Rows are defined as rows in each layer of depth

    #Code:
    row_min = Y.min(axis=2).repeat(3).reshape(3,3,3)
    print(f"row_min:\n{row_min}")
    print(Y-row_min)
```
## Part 5
### 5a
Claim: $\nabla x(x^Ta) = \nabla x(a^Tx) = a$

|Proof: $x^Ta = a^Tx$|
|:---:|
$x^Ta = x_1a_1 + ... + x_na_n = \sum x_ia_i$
$a^Tx = a_1x_1 + ... + a_nx_n =\sum a_ix_i$

$\therefore$ since $x_ia_i = a_ix_i$, these values are equal.

|Proof: $\nabla x(x^Ta) = a$|
|:-----:|
Let $ f= x^Ta$ 
$f = x_1a_1 + ... + x_na_n$ 
$\frac{\partial f}{\partial x_1} = 1a_1 + 0a_2 + ... 0a_n$
$\frac{\partial f}{\partial x_2} = 0a_1 + 1a_2 + ... 0a_n$
⋮
$\frac{\partial f}{\partial x_n} = 0a_1 + 0a_2 + ... 1a_n$

$\therefore \nabla xf = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ ⋮ \\ \frac{\partial f}{\partial x_n} \end{bmatrix} = \begin{bmatrix} a_1 \\ ⋮ \\ a_n \end{bmatrix} = a$

### 5b




