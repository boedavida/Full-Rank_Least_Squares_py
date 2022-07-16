# test_la.py by David Boe

import pytest
import numpy as np
from householder import house
from qr import qr
from LS_Householder import LS_Householder

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
    """Test of A = Q @ R"""
    sh = np.shape(A)
    m = sh[0]
    n = sh[1]
    QR = qr(A, alg='house')
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
    # Unit test
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
    QR = qr(A, alg='house')
    sh = np.shape(A)
    m = sh[0]
    n = sh[1]
    # Forward accumulation of Q: Multiply Householder matrices to get Q = H1*H2*H3* ... Hn
    Q = np.eye(m)
    for j in range(0,n):
        b = 2./(1. + np.linalg.norm(QR[j+1:m,j],ord=2)**2)
        vj = np.zeros(m)
        vj[j:m] = np.hstack((np.array(1.),QR[j+1:m,j]))
        Q = Q @ (np.eye(m) - b*np.outer(vj,vj))
    assert np.allclose(Q @ np.transpose(Q), np.eye(m), rtol=np.finfo(float).eps) == True

@pytest.mark.parametrize(
    "A",
    [   (np.random.normal(loc=0.0, scale=5.0, size=(4,3))),
        (np.random.normal(loc=0.0, scale=5.0, size=(12,10))),
        (np.random.normal(loc=0.0, scale=5.0, size=(9,7)))
    ]
)
def test_LS_Householder_1(A):
    sh = np.shape(A)
    m = sh[0]
    n = sh[1]
    x = np.random.normal(loc=0.0, scale=3.0, size=n)
    b = A @ x 
    x_LS = LS_Householder(A, b)
    # Unit test
    assert np.allclose( x_LS, x, rtol=np.finfo(float).eps) == True

@pytest.mark.parametrize(
    "A",
    [   (np.random.normal(loc=0.0, scale=5.0, size=(4,3))),
        (np.random.normal(loc=0.0, scale=5.0, size=(12,10))),
        (np.random.normal(loc=0.0, scale=5.0, size=(9,7)))
    ]
)
def test_LS_Householder_2(A):
    """Test LS_Householder against numpy.linalg.lstsq"""
    sh = np.shape(A)
    m = sh[0]
    n = sh[1]
    x = np.random.normal(loc=0.0, scale=3.0, size=n)
    b = A @ x 
    # Numpy solution
    x_LS_np, res, rnk, s = np.linalg.lstsq(A, b, rcond=None)
    x_LS = LS_Householder(A, b)
    # Unit test
    assert np.allclose( x_LS, x_LS_np, rtol=np.finfo(float).eps) == True
