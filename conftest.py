# conftest.py

import pytest
import numpy as np

@pytest.fixture() 
def qr_test_case_1(): 
    return np.random.normal(loc=0.0, scale=5.0, size=(4,3))

@pytest.fixture() 
def qr_test_case_2(): 
    return np.random.normal(loc=0.0, scale=5.0, size=(12,10))

@pytest.fixture() 
def qr_test_case_3(): 
    return np.random.normal(loc=0.0, scale=5.0, size=(9,7))