# Full-Rank Least Squares
Linear algebra functions implemented in Python 3 to solve the full-rank least squares problem by QR factorization with Householder reflections. The full-rank least squares problem is the problem of finding an n-vector x with real entries such that Ax = b, where the matrix A is m x n with real entries and with m >= n and has rank n, and b is an m-vector with real entries. This is an overdetermined system of linear equations. The algorithms are those from Golub and Van Loan, Matrix Computations. 

Requires Python 3.6 or greater (there are f-strings), NumPy and pytest.

LS_Householder.py is the executive file. It generates a linear system Ax = b, runs the algorithm that determines the least squares solution, and tests the result against the truth. 
To run unit test, the command is $ pytest test_ls.py

Next step: The next step, which is under development, is the addition of perturbations and comparision of the result with theory. 
