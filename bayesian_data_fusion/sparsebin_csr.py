import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait


class SparseBinMatrixCSR:
    def __init__(self, m, n, col_ind, row_ptr):
        """Initializes a SparseBinMatrixCSR.

        Parameters:
        m: int, Number of rows.
        n: int, Number of columns.
        col_ind: np.array of int32, Column indices of non-zero elements.
        row_ptr: np.array of int32, Row pointers.
        """
        self.m = m  # Number of rows
        self.n = n  # Number of columns
        self.col_ind = col_ind  # Column indices for non-zero elements
        self.row_ptr = row_ptr  # Row pointers

    @classmethod
    def from_rows_cols(cls, rows, cols):
        """Constructs a SparseBinMatrixCSR from row and column indices.

        Parameters:
        rows: np.array of int32, Row indices of non-zero elements.
        cols: np.array of int32, Column indices of non-zero elements.

        Returns:
        SparseBinMatrixCSR: An instance of SparseBinMatrixCSR.
        """
        # Sort row indices and get sorted indices
        rsorted = np.argsort(rows, kind="stable")

        # Determine the matrix dimensions
        m = int(np.max(rows))
        n = int(np.max(cols))

        # Initialize row pointers
        row_ptr = np.zeros(m + 1, dtype=np.int32)
        row_ptr[:] = len(rows) + 1

        # Sort rows based on sorted indices
        rows_sorted = rows[rsorted]

        # Populate row_ptr with cumulative counts for each row
        prev = np.int32(0)
        for i in range(len(rows)):
            while rows_sorted[i] > prev:
                prev += 1
                row_ptr[prev] = i

        # Return a new instance of SparseBinMatrixCSR with sorted columns
        return cls(m, n, cols[rsorted].astype(np.int32), row_ptr)

    def size(self, dim=None):
        """Returns the size of the SparseBinMatrixCSR.

        Parameters:
        dim: int, Optional dimension (1 for rows, 2 for columns).

        Returns:
        tuple or int: If dim is None, returns a tuple (m, n); if dim is specified,
        returns the size along that dimension.
        """
        if dim is None:
            return self.m, self.n
        elif dim == 1:
            return self.m
        elif dim == 2:
            return self.n
        else:
            raise ValueError("Dimension must be 1 or 2.")

    def __str__(self):
        """Returns a string representation of the SparseBinMatrixCSR.

        Returns:
        str: Description of the SparseBinMatrixCSR.
        """
        return f"{self.m} x {self.n} binary sparse matrix (CSR) with {len(self.col_ind)} entries."

    def __repr__(self):
        return self.__str__()


class ParallelBinCSR:
    def __init__(self, m, n, pids, csrs, mranges, blocksize):
        """Initializes a ParallelBinCSR for parallel processing of SparseBinMatrixCSR
        blocks.

        Parameters:
        m: int, Number of rows.
        n: int, Number of columns.
        pids: list of int64, Process IDs for parallel execution.
        csrs: list, References to SparseBinMatrixCSR or related structures for each process.
        mranges: list of ranges, Step ranges for block assignments in each process.
        blocksize: int64, Size of each block for parallel processing.
        """
        self.m = m  # Number of rows
        self.n = n  # Number of columns
        self.pids = pids  # Process IDs for parallel execution
        self.csrs = csrs  # References to SparseBinMatrixCSR structures or similar
        self.mranges = mranges  # Step ranges for each process
        self.blocksize = blocksize  # Block size for processing

    @classmethod
    def from_rows_cols(cls, rows, cols, pids, blocksize=64):
        """Constructs a ParallelBinCSR from row and column indices and process IDs.

        Parameters:
        rows: np.array of int32, Row indices of non-zero elements.
        cols: np.array of int32, Column indices of non-zero elements.
        pids: list of int, Process IDs for parallel execution.
        blocksize: int, Size of each block for parallel processing (default: 64).

        Returns:
        ParallelBinCSR: An instance of ParallelBinCSR.
        """
        csr = SparseBinMatrixCSR.from_rows_cols(rows, cols)
        npids = len(pids)
        ranges = [range(1 + i * blocksize, csr.m + 1, npids * blocksize) for i in range(npids)]

        # Initialize ParallelBinCSR with placeholders for csrs (list of futures)
        pcsr = cls(csr.m, csr.n, pids, [], ranges, blocksize)

        # Use ProcessPoolExecutor to spawn parallel tasks
        with ProcessPoolExecutor() as executor:
            for pid in pids:
                ref = executor.submit(lambda: csr)
                pcsr.csrs.append(ref)

        return pcsr


def A_mul_B_range(y, A_or_csr, x, row_or_mrange, blocksize):
    """Helper function to multiply a range of rows of the CSR matrix with vector x and
    store the result in y.

    Parameters:
    y: np.array, The result vector to be modified in place.
    A_or_csr: SparseBinMatrixCSR, The CSR matrix (can be named either A or csr).
    x: np.array, The vector to multiply with A_or_csr.
    row_or_mrange: range, The range of rows to process (can be named either row_range or mrange).
    blocksize: int, The size of each processing block.

    Raises:
    ValueError: If the dimensions of A_or_csr, x, or y do not match.
    """
    if A_or_csr.n != len(x):
        raise ValueError(f"A_or_csr.n={A_or_csr.n} must equal length(x)={len(x)}")
    if A_or_csr.m != len(y):
        raise ValueError(f"A_or_csr.m={A_or_csr.m} must equal length(y)={len(y)}")

    row_ptr = A_or_csr.row_ptr
    col_ind = A_or_csr.col_ind

    # Perform multiplication within the specified row range and block size
    for b0 in row_or_mrange:
        for row in range(b0, min(b0 + blocksize, A_or_csr.m)):
            tmp = 0
            for i in range(row_ptr[row], row_ptr[row + 1]):
                tmp += x[col_ind[i]]
            y[row] = tmp


def A_mul_B_part_ref(y, Aref, x, row_range, blocksize):
    """Retrieves the SparseBinMatrixCSR from a Future and applies a partial in-place
    matrix-vector multiplication on y.

    Parameters:
    y: np.array, The result vector to be modified in place.
    Aref: Future, A future containing the SparseBinMatrixCSR matrix.
    x: np.array, The vector to multiply with A.
    row_range: range, Range of rows to process.
    blocksize: int, Size of each processing block.
    """
    A = Aref.result()  # Retrieve the SparseBinMatrixCSR from the future
    A_mul_B_range(y, A, x, row_range, blocksize)


def A_mul_B(y, A, x):
    """Performs in-place sparse matrix-vector multiplication y = A * x.

    If A is a SparseBinMatrixCSR, performs sequential matrix-vector multiplication.
    If A is a ParallelBinCSR, performs parallel matrix-vector multiplication.

    Parameters:
    y: np.array, The result vector to be modified in place.
    A: SparseBinMatrixCSR or ParallelBinCSR, The sparse matrix structure.
    x: np.array, The vector to multiply with A.

    Raises:
    ValueError: If dimensions of A and x, or A and y, do not match.
    """
    if A.n != len(x):
        raise ValueError(f"A.n={A.n} must equal length(x)={len(x)}")
    if A.m != len(y):
        raise ValueError(f"A.m={A.m} must equal length(y)={len(y)}")

    if isinstance(A, SparseBinMatrixCSR):
        _A_mul_B_sparse(y, A, x)
    elif isinstance(A, ParallelBinCSR):
        _A_mul_B_parallel(y, A, x)
    else:
        raise TypeError(
            "Unsupported matrix type for A. Expected SparseBinMatrixCSR or ParallelBinCSR."
        )


def _A_mul_B_sparse(y, A, x):
    """Performs in-place sparse matrix-vector multiplication for SparseBinMatrixCSR."""
    row_ptr = A.row_ptr
    col_ind = A.col_ind

    # Perform sparse matrix-vector multiplication
    for row in range(A.m):
        tmp = 0  # Initialize temporary sum variable
        for i in range(row_ptr[row], row_ptr[row + 1]):
            tmp += x[col_ind[i]]
        y[row] = tmp


def _A_mul_B_parallel(y, A, x):
    """Performs in-place parallel matrix-vector multiplication for ParallelBinCSR."""

    def A_mul_B_part(y, csr_future, x, mrange, blocksize):
        csr = csr_future.result()  # Retrieve CSR matrix from future
        A_mul_B_range(y, csr, x, mrange, blocksize)

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor() as executor:
        futures = []
        for p in range(len(A.pids)):
            csr_future = A.csrs[p]  # Reference to the CSR matrix future for this process
            mrange = A.mranges[p]
            # Schedule the partial multiplication as a parallel task
            futures.append(executor.submit(A_mul_B_part, y, csr_future, x, mrange, A.blocksize))

        # Wait for all tasks to complete
        wait(futures)


def A_mul_B_range(y, csr, x, row_range, blocksize):
    """Helper function to multiply a range of rows of the CSR matrix with vector x and
    store the result in y."""
    row_ptr = csr.row_ptr
    col_ind = csr.col_ind

    for b0 in row_range:
        for row in range(b0, min(b0 + blocksize, csr.m)):
            tmp = 0
            for i in range(row_ptr[row], row_ptr[row + 1]):
                tmp += x[col_ind[i]]
            y[row] = tmp
