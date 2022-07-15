# qr.py by David Boe

import numpy as np
from householder import house

def qr(A, alg='house'):
    """QR decomposition of m x n matrix A, with m >= n
    This function explicitly outputs an orthogonal Q 
    and an upper triangular R. This algorithm is based on a
    more efficient algorithm that returns the Householder 
    vectors in the lower triangular components of R and 
    takes 2*n^2*(m - n/3) flops"""

    # Check that A is a numpy array
    if type(A).__name__ != 'ndarray':
        raise TypeError('A must be a numpy ndarray')

    # Check that entries in A are real numbers (floats) and not ints
    if A.dtype != 'float64' and A.dtype != 'float32':
        raise TypeError('Entries in A must be real numbers (floats)')

    # Get the shape of A
    sh = np.shape(A)

    # Error checking
    if np.size(A) == 0:
        raise ValueError('A is empty')
    if sh[0] < sh[1]:
        raise ValueError('A must have at least as many rows as columns')
    
    if alg == 'house':
        m = sh[0]
        n = sh[1]
        R = np.copy(A)
        Q = np.eye(m)
        for j in range(0,n):
            v, b = house(R[j:m,j])
            I = np.eye(np.size(v))
            R[j:m,j:n] = (I - b*np.outer(v,v)) @ R[j:m,j:n] # @ is shorthand for np.matmul()
            
            if j < m:
                # Store the components of the Householder vectors in the lower triangule part of R
                # R[j+1:m,j] = v[1:(m-1-j+1)] 
                # Overwrite lower triangular components with zeros 
                R[j+1:m,j] = np.zeros(m-j-1, dtype=float) 

            # Forward accumulation of Q: Multiply Householder matrices to get Q = H1*H2*H3* ... Hn
            vj = np.zeros(m)
            vj[j:m] = v
            H = (np.eye(m) - b*np.outer(vj,vj))
            Q = Q @ H
        return Q, R
            
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
    Q, R = qr(A, alg='house')
    
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
        print(f"Orthogonality of Q: PASSED")
    else:
        print(f"Orthogonality of Q: FAILED")
    
    # Test of upper triangularity of R
    for p in range(0,n):
        if np.allclose(R[p+1:m,p], 0, rtol=np.finfo(float).eps) == False:
            print(f"Upper triangularity of R: FAILED\n")
            break
    else:
        print(f"Upper triangularity of R: PASSED\n")


if __name__ == "__main__":
    main()
