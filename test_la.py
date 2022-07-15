# test_la.py by David Boe

import pytest
import numpy as np
from householder import house
from qr import qr

# Make test cases for test_house
@pytest.mark.parametrize(
    "x",
    [
        (np.random.normal(loc=0.0, scale=5.0, size=np.random.randint(2,10))),  
        (np.random.normal(loc=0.0, scale=5.0, size=np.random.randint(2,10))),  
        (np.random.normal(loc=0.0, scale=5.0, size=np.random.randint(2,10)))
    ]
)
def test_house(x):
    """Tests house() in householder.py"""
    
    # Call house()
    v, b = house(x)

    # Computations necessary for the evalution
    m = np.size(x)
    Im = np.eye(m)
    e1 = Im[:,0]
    P = Im - b*np.outer(v,v)
    Im = np.eye(m)
    e1 = Im[:,0]

    # Evaluation
    assert np.allclose(np.matmul(P,x), np.linalg.norm(x)*e1, rtol=np.finfo(float).eps) == True

# Make test cases for test_qr_*
@pytest.mark.parametrize(
    "A",
    [   (np.random.normal(loc=0.0, scale=5.0, size=(4,3))),
        (np.random.normal(loc=0.0, scale=5.0, size=(12,10))),
        (np.random.normal(loc=0.0, scale=5.0, size=(9,7)))
    ]
)
def test_qr_AQR(A):
    """ Test of A = Q @ R"""
    Q, R = qr(A, alg='house')
    assert np.allclose(A, Q @ R, rtol=np.finfo(float).eps) == True

@pytest.mark.parametrize(
    "A",
    [   (np.random.normal(loc=0.0, scale=5.0, size=(4,3))),
        (np.random.normal(loc=0.0, scale=5.0, size=(12,10))),
        (np.random.normal(loc=0.0, scale=5.0, size=(9,7)))
    ]
)
def test_qr_Q_orth(A):
    """Test of orthogonality of Q"""
    Q, R = qr(A, alg='house')
    sh = np.shape(A)
    m = sh[0]
    assert np.allclose(Q @ np.transpose(Q), np.eye(m), rtol=np.finfo(float).eps) == True

@pytest.mark.parametrize(
    "A",
    [   (np.random.normal(loc=0.0, scale=5.0, size=(4,3))),
        (np.random.normal(loc=0.0, scale=5.0, size=(12,10))),
        (np.random.normal(loc=0.0, scale=5.0, size=(9,7)))
    ]
)
def test_qr_R_upper(A):
    """Test of upper triangularity of R"""
    Q, R = qr(A, alg='house')
    sh = np.shape(A)
    m = sh[0]
    n = sh[1]
    rslt = True
    for p in range(0,n):
        if np.allclose(R[p+1:m,p], 0, rtol=np.finfo(float).eps) == False:
            rslt = False
            break
    assert rslt == True
