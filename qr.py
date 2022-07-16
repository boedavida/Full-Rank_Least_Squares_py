# qr.py by David Boe

import numpy as np
from householder import house

def qr(A: np.array(float), alg='house') -> np.array(float):
    """QR decomposition of m x n matrix A, with m >= n
    This function is an implimentation of Algorithm 5.2.1 
    (Householder QR) in Golub and Van Loan, Matrix 
    Computations. The algorithm that overwrites A with
    upper triangular R takes 2*n^2*(m - n/3) flops and
    storesHhouseholder vectors in the lower triangular
    components. This algorithm takes 2*n^2*(m - n/3) flops."""

    # Check that A is a numpy array
    if type(A).__name__ != 'ndarray':
        raise TypeError('A must be a numpy ndarray')

    # Check that entries in A are real numbers (floats) and not ints
    if A.dtype != np.float64 and A.dtype != np.float32:
        raise TypeError('Entries in A must be real numbers (floats)')

    # Get the shape of A
    sh = np.shape(A)

    # Error checking
    if np.size(A) == 0:
        return np.array([])
    if sh[0] < sh[1]:
        raise ValueError('A must have at least as many rows as columns')
    
    if alg == 'house':
        m = sh[0]
        n = sh[1]
        QR = np.copy(A)
        for j in range(0,n):
            v, b = house(QR[j:m,j])
            I = np.eye(np.size(v))
            QR[j:m,j:n] = (I - b*np.outer(v,v)) @ QR[j:m,j:n] # @ is shorthand for np.matmul()
            # Store the components of the Householder vectors in the lower triangular part of R
            if j < m:
                QR[j+1:m,j] = v[1:(m-1-j+1)] 
        return QR
            
    elif alg == 'givens':
        raise ValueError('alg either not specified correctly or not supported')
    else:
        raise ValueError('alg either not specified correctly or not supported')


def main():
    # Generate a random rectangular m x n matrix such that m >= n
    m = 4
    n = 3
    mn = 0.0
    std = 3.0
    A = np.random.normal(loc=mn, scale=std, size=(m,n))

    # QR decomposition of A
    QR = qr(A, alg='house')

    # Forward accumulation of Q: Multiply Householder matrices to get Q = H1*H2*H3* ... Hn
    # Takes 4*(m^2*n - m*n^2 + n^3/3) to accumulate Householder vectors
    Q = np.eye(m)
    for j in range(0,n):
        b = 2./(1. + np.linalg.norm(QR[j+1:m,j],ord=2)**2)
        vj = np.zeros(m)
        vj[j:m] = np.hstack((np.array(1.),QR[j+1:m,j]))
        Q = Q @ (np.eye(m) - b*np.outer(vj,vj))
    
    # Form R, upper triangular for verification 
    R = np.copy(QR)
    for j in range(0,n):
        R[j+1:m,j] = np.zeros(m-j-1)

    print(f"\nA = \n{A}")
    print(f"\nQ = \n{Q}")
    print(f"\nR = \n{R}")
    
    # Test of A = Q @ R
    if np.allclose(A, Q @ R, rtol=np.finfo(float).eps):
        print(f"\nA = Q @ R test: PASSED")
    else:
        print(f"\nA = Q @ R test: FAILED")
    
    # Test of orthogonality of Q
    if np.allclose( Q @ np.transpose(Q), np.eye(m), rtol=np.finfo(float).eps):
        print(f"Orthogonality of Q test: PASSED")
    else:
        print(f"Orthogonality of Q test: FAILED")
    

if __name__ == "__main__":
    main()
    