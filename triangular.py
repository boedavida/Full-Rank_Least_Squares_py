# triangular.py

import numpy as np

def backsubstitution(U: np.array(float), b:np.array(float)) -> np.array(float):
    """Solves the linear system Ux = b for upper
    triangular n x n matrix U by back susbstitution.
    Algorithm 3.1.2 in Golub and Van Loan, Matrix
    Computations. Upper triangularity of U is assumed
    but not enforced as this state is required for 
    working with QR and LS_Householder algorithms in
    the QR least squares problem. Lower triangular 
    components are ignored in this algorithm."""

    # Error checking
    # Check that A is a numpy array
    if type(U).__name__ != 'ndarray':
        raise TypeError('A must be a numpy ndarray')

    # Check that entries in A are real numbers (floats) and not ints
    if U.dtype != 'float64' and U.dtype != 'float32':
        raise TypeError('Entries in A must be real numbers (floats)')

    # Get the shape of A
    sh = np.shape(U)
    m = sh[0]
    n = sh[1]   

    # Get the size of b
    sb = np.size(b)

    if np.size(U) == 0:
        return np.array([])
    if m != n:
        raise TypeError('U must be square')
    if sb != n:
        raise TypeError('b is not of size n')       
    #Test for upper triangularity
    for p in range(0,n):
        if np.allclose(U[p+1:n,p], 0, rtol=np.finfo(float).eps) == False:
            print(f'\nWARNING (backsubstitution): U is not upper triangular') 
            break

    # Back substitution algorithm. Overwrites b
    b[n-1] = b[n-1]/U[n-1,n-1]
    for i in range(n-2,-1,-1):
        b[i] = (b[i] - np.dot(U[i,i+1:n],b[i+1:n]))/U[i,i]
    # Outputs the solution x, which is stored in b
    return b 

def main():
    # Generate a random rectangular n x n matrix and make it upper triangular
    n = 3
    mn = 0.0
    std = 3.0
    A = np.random.normal(loc=mn, scale=std, size=(n,n))
    U = np.triu(A, k=0)

    # Generate a n-vector for the true solution
    x0 = np.random.normal(loc=mn, scale=std, size=n)

    # Compute b
    b = U @ x0

    # Solve for Ux = b for x by back substitution
    x = backsubstitution(U, b)

    # Unit test
    if np.allclose( x, x0, rtol=np.finfo(float).eps):
        print(f"\nBack substitution test: PASSED\n")
    else:
        print(f"\nBack substitution test: FAILED\n")

if __name__ == "__main__":
    main()