# test_la.py

import pytest
import numpy as np
from householder import house

# Make test cases for test_house
@pytest.mark.parametrize(
    "x, expected",
    [
        (np.random.normal(loc=0.0, scale=5.0, size=np.random.randint(2,10)), True),  
        (np.random.normal(loc=0.0, scale=5.0, size=np.random.randint(2,10)), True),  
        (np.random.normal(loc=0.0, scale=5.0, size=np.random.randint(2,10)), True)
    ]
)
def test_house(x, expected):
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
    assert np.allclose(np.matmul(P,x), np.linalg.norm(x)*e1, rtol=np.finfo(float).eps) == expected