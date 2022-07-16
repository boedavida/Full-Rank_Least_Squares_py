# LS_Householder.py
"""Solution of full-rank least squares problem by
QR factorization"""

from cmath import nan
import numpy as np
from householder import house
from qr import qr
from triangular import backsubstitution

def LS_Householder(A, b):
    """Householder least squares solution for the
    full rank least squares problem for Ax = b with 
    m x n matrix A, m >= n, rank(A) = n, and m-vector b"""
    # Get the shape of A
    sh = np.shape(A)
    m = sh[0]
    n = sh[1]
    # QR factorization to overright A with QR factorization
    R = qr(A, alg='house')
    for j in range(0,n):
        v = np.hstack((np.array(1.),R[j+1:m,j]))
        beta = 2./(np.dot(v,v))
        b[j:m] = b[j:m] - beta*np.dot(v,b[j:m])*v
    # Solve R[0:n, 0:n] * x_LS = b[0:n] by back subsitution
    x = backsubstitution(R[0:n, 0:n], b[0:n])
    return x

def main():
    # Generate a random rectangular m x n matrix such that m >= n
    m = 4
    n = 3
    mn = 0.0
    std = 3.0
    A = np.random.normal(loc=mn, scale=std, size=(m,n))

    # Generate a n-vector vector b
    x = np.random.normal(loc=mn, scale=std, size=n)
    b = A @ x #+ np.random.normal(loc=mn, scale=1e-3, size=m) # Add random noise to measured values

    # Numpy solution
    x_LS_np, res, rnk, s = np.linalg.lstsq(A, b, rcond=None)
    x_LS = LS_Householder(A, b)

    # Unit test
    if np.allclose( x_LS, x, rtol=np.finfo(float).eps):
        print(f"\nLeast Squares householder test #1: PASSED")
    else:
        print(f"\nLeast Squares householder test #1: FAILED")
    
    # Unit test
    if np.allclose( x_LS, x_LS_np, rtol=np.finfo(float).eps):
        print(f"Least Squares householder test #2: PASSED\n")
    else:
        print(f"Least Squares householder test #2: FAILED\n")

    print(x)
    print(x_LS_np)
    print(x_LS)

if __name__ == "__main__":
    main()
