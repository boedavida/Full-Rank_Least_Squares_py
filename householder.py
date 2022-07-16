# householder.py by David Boe

import numpy as np

def house(x: np.array(float)) -> tuple[np.array(float), float]:
    """ This function computes the Householder vector based on 
    Algorithm 5.1.1 in Golub and Van Loan, Matrix Comptutations.
    Given x is an element of R^m, this function computes v, an element of R^m,
    with v[0] (zero-based indexing) and beta, a real number, such that 
    P = I - beta*v*v^T is orthogonal and P*x = || x ||_2 *e_1.
    This algorithm takes about 3*m flops, so it's O(m)."""

    # Check that x is a numpy array
    if type(x).__name__ != 'ndarray':
        raise TypeError('x must be a numpy ndarray')

    # Check that entries in a are real numbers (floats) and not ints
    if x.dtype != 'float64' and x.dtype != 'float32':
        raise TypeError('Entries in x must be real numbers (floats)')
    
    # Get the number of elements in x
    m = np.size(x)

    # Error checking
    if m == 0:
        return np.array([])
    if np.ndim(x) > 1:
        raise ValueError('x must be a vector')
    
    sigma = np.dot(x[1:m],x[1:m])
    v = np.hstack((np.array(1.),x[1:m]))
    if np.allclose(sigma, 0, atol=1e-16) and x[0] >= 0:
        beta = 0
    elif np.allclose(sigma, 0, atol=1e-16) and x[0] < 0:
        beta = -2
    else:
        mu = np.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma/(x[0] + mu)
        beta = 2*v[0]**2/(sigma + v[0]**2)
        v = v/v[0]

    return v, beta


def main():
    m = 3
    mn = 0.0
    std = 3.0
    x = np.random.normal(loc=mn, scale=std, size=m)
    v, b = house(x)
    Im = np.eye(m)
    e1 = Im[:,0]

    # Test
    P = Im - b*np.outer(v,v)
    if np.allclose(np.matmul(P,x), np.linalg.norm(x)*e1, rtol=np.finfo(float).eps):
        print(f"\nPASSED\n")
    else:
        print(f"\nFAILED\n")


if __name__ == "__main__":
    main()