from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.sparse import csr_matrix
import os


class ParallelSparseMatrix:
    """
    A class to represent a parallel sparse matrix.

    Attributes:
    F : Any
        The sparse matrix.
    refs : list
        A list of references for parallel execution.
    procs : list
        A list of process IDs.
    """

    def __init__(self, F, procs):
        self.F = F
        self.refs = []
        self.procs = procs

def psparse(F, procs):
    """
    Creates a ParallelSparseMatrix by distributing the workload across specified processes.

    Parameters:
    F : Any
        The sparse matrix to be parallelized.
    procs : list
        A list of process IDs for parallel execution.

    Returns:
    ParallelSparseMatrix
        An instance of ParallelSparseMatrix with references to the distributed processes.
    """
    # In a real implementation, you'd want to spawn processes and fetch the sparse matrix
    parallel_matrix = ParallelSparseMatrix(F, procs)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch, F, proc): proc for proc in procs}
        for future in as_completed(futures):
            proc = futures[future]
            parallel_matrix.refs.append(future.result())  # Fetch the result and append it to refs

    return parallel_matrix

class ParallelSparseMatrix:
    def __init__(self, F, refs, procs):
        self.F = F          # The sparse matrix
        self.refs = refs    # References for parallel execution
        self.procs = procs  # Process IDs

    def is_empty(self):
        """ Checks if the sparse matrix is empty. """
        return self.F.size == 0

    def shape(self):
        """ Returns the shape of the sparse matrix. """
        return self.F.shape

    def eltype(self):
        """ Returns the element type of the sparse matrix. """
        return self.F.dtype

def At_mul_B(A, B):
    """
    Overloaded function to handle various types of matrix operations.

    Parameters:
    - A: The first matrix, can be ParallelSparseMatrix or a standard numpy array.
    - B: The second matrix, can be ParallelSparseMatrix or a standard numpy array.

    Returns:
    - Result of A.T @ B for the different types of matrices.
    """
    if isinstance(A, ParallelSparseMatrix) and isinstance(B, np.ndarray):
        # Case: ParallelSparseMatrix and numpy array
        return pmult(A.F.shape[1], A.refs, B, A.procs, imult)

    elif isinstance(A, ParallelSparseMatrix) and isinstance(B, ParallelSparseMatrix):
        # Case: Both A and B are ParallelSparseMatrix
        return At_mul_B(A.F, B.F)

    elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        # Case: Both A and B are standard numpy arrays
        return np.dot(A.T, B)

    else:
        raise TypeError("Unsupported types for At_mul_B: {}, {}".format(type(A), type(B)))

class SparseMatrixCSR:
    def __init__(self, csc):
        self.csc = csc  # Store the CSC representation

    @classmethod
    def from_csc(cls, csc):
        """Create a SparseMatrixCSR from a CSC matrix."""
        return cls(csc)

    @classmethod
    def from_rows_cols_vals(cls, rows, cols, vals):
        """Create a SparseMatrixCSR from row, column, and value arrays."""
        return cls(csr_matrix((vals, (rows, cols))).tocsc())

    def at_mul_b(self, u):
        """Computes A^T * u."""
        return self.csc.T @ u

    def ac_mul_b(self, u):
        """Computes A * u (same as A^T * u for CSR)."""
        return self.csc @ u

    def __mul__(self, u):
        """Overload * operator for A * u."""
        return self.at_mul_b(u)

    def is_empty(self):
        """Check if the matrix is empty."""
        return self.csc.nnz == 0

    def eltype(self):
        """Get the element type of the sparse matrix."""
        return self.csc.dtype

    def size(self):
        """Return the shape of the sparse matrix."""
        return self.csc.shape

    def __getitem__(self, d):
        """Get size based on the dimension index."""
        if d > 2:
            return 1
        return self.size()[d - 1]  # Adjust for zero-based index

def imult(Fref, u):
    """Performs the operation A^T * u where A is fetched from Fref."""
    return At_mul_B(Fref.result(), u)

def mult(Fref, u):
    """Performs the operation A * u where A is fetched from Fref."""
    Flocal = Fref.result()
    if Flocal.shape[1] != u.shape[0]:
        raise ValueError(f"#columns of F({Flocal.shape[1]}) must equal number of rows of U({u.shape[0]}).")
    return Flocal @ u

def nextidx(index):
    """Produces the next work item from the queue."""
    idx = index[0]
    index[0] += 1
    return idx

def pmult(nrows, Frefs, U, procs, mfun):
    """
    Parallel multiplication function.

    Parameters:
    nrows : int
        The number of rows in the output.
    Frefs : list
        List of future references for the feature matrices.
    U : numpy.ndarray
        The matrix to multiply with.
    procs : list
        List of process identifiers (or threads).
    mfun : function
        The function to be executed remotely.

    Returns:
    results : numpy.ndarray
        The result of the parallel multiplication.
    """
    np = len(procs)  # Number of processes available
    n = U.shape[1]
    results = np.zeros((nrows, n))
    index = [1]  # Use a list to maintain state

    # Synchronize and perform parallel processing
    for p in range(len(procs)):
        if procs[p] != os.getpid() or np == 1:  # Check if the current process is not the same
            # Start an asynchronous task
            while True:
                idx = nextidx(index)
                if idx > n:
                    break
                results[:, idx - 1] = mfun(Frefs[p], U[:, idx - 1])  # Adjust for zero-based indexing

    return results
