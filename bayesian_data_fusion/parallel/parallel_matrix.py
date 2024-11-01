import concurrent.futures
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Array, Pool, Semaphore

import numpy as np


class SparseBinMatrix:
    def __init__(self, m, n, rows, cols):
        """Initializes a SparseBinMatrix instance.

        Parameters:
        m : int
            The number of rows.
        n : int
            The number of columns.
        rows : list of int
            The row indices of non-zero entries.
        cols : list of int
            The column indices of non-zero entries.
        """
        if len(rows) != len(cols):
            raise ValueError("length(rows) must equal length(cols)")

        self.m = m
        self.n = n
        self.mrange = range(min(rows), max(rows) + 1)  # Python range is exclusive at the end
        self.nrange = range(min(cols), max(cols) + 1)
        self.rows = rows
        self.cols = cols

    def size(self):
        return self.m, self.n

    def isempty(self):
        return self.m == 0 or self.n == 0


def create_sparse_bin_matrix(m, n, rows, cols):
    """Helper function to create a SparseBinMatrix."""
    return SparseBinMatrix(m, n, rows, cols)


def create_sparse_bin_matrix_from_vectors(rows, cols):
    """Create a SparseBinMatrix using row and column vectors."""
    return SparseBinMatrix(max(rows), max(cols), rows, cols)


def create_sparse_bin_matrix_from_int64(rows, cols):
    """Create a SparseBinMatrix using int64 vectors for rows and columns."""
    rows_int32 = np.array(rows, dtype=np.int32)
    cols_int32 = np.array(cols, dtype=np.int32)
    return SparseBinMatrix(max(rows_int32), max(cols_int32), rows_int32, cols_int32)


def get_index(sbm, rows):
    """Retrieves a subset of the SparseBinMatrix based on boolean row indices.

    Parameters:
    sbm : SparseBinMatrix
        The sparse bin matrix from which to extract data.
    rows : list of bool
        Boolean list indicating which rows to select.

    Returns:
    SparseBinMatrix : A new SparseBinMatrix instance with selected rows.
    """
    if len(rows) != sbm.m:
        raise ValueError("length(rows) must equal size(sbm, 1)")

    idx = [i for i, v in enumerate(rows) if v]
    colidx = [sbm.cols[i] for i in sbm.rows if i in idx]
    rsum = np.cumsum(rows)
    rowidx = rsum[sbm.rows[idx]]

    return SparseBinMatrix(
        sum(rows), sbm.n, np.arange(1, sum(rows) + 1, dtype=np.int32), sbm.nrange, rowidx, colidx
    )


def get_index_with_cols(sbm, rows, cols):
    """Retrieves a subset of the SparseBinMatrix based on boolean row indices and a
    column selector.

    Parameters:
    sbm : SparseBinMatrix
        The sparse bin matrix from which to extract data.
    rows : list of bool
        Boolean list indicating which rows to select.
    cols : slice or range
        Specifies which columns to select.

    Returns:
    SparseBinMatrix : A new SparseBinMatrix instance with selected rows.
    """
    return get_index(sbm, rows)


def get_index_with_range(sbm, rows, col_range):
    """Retrieves a subset of the SparseBinMatrix based on boolean row indices and a
    column range.

    Parameters:
    sbm : SparseBinMatrix
        The sparse bin matrix from which to extract data.
    rows : list of bool
        Boolean list indicating which rows to select.
    col_range : range
        Specifies which columns to select.

    Returns:
    SparseBinMatrix : A new SparseBinMatrix instance with selected rows.
    """
    return get_index(sbm, rows)


class ParallelLogic:
    def __init__(
        self, mblocks, nblocks, mblock_order, nblock_order, localm, localn, tmp=None, sems=None
    ):
        """Initializes a ParallelLogic instance.

        Parameters:
        mblocks : list of range
            The ranges of rows for the blocks.
        nblocks : list of range
            The ranges of columns for the blocks.
        mblock_order : list of int
            The order of the row blocks.
        nblock_order : list of int
            The order of the column blocks.
        localm : list of float
            Local vector for rows.
        localn : list of float
            Local vector for columns.
        tmp : SharedVector (optional)
            Temporary vector for calculations.
        sems : list of SharedArray (optional)
            List of semaphores for synchronization.
        """
        self.mblocks = mblocks
        self.nblocks = nblocks
        self.mblock_order = mblock_order
        self.nblock_order = nblock_order
        self.localm = localm
        self.localn = localn
        self.tmp = tmp
        self.sems = sems


def nonshared(A):
    """Creates a non-shared instance of ParallelLogic."""
    return ParallelLogic(A.mblocks, A.nblocks, A.mblock_order, A.nblock_order, A.localm, A.localn)


class ParallelSBM:
    def __init__(
        self,
        m,
        n,
        pids,
        weights=None,
        sbms=None,
        logic=None,
        numblocks=None,
        tmp=None,
        sh1=None,
        sh2=None,
        sems=None,
    ):
        """Initializes a ParallelSBM instance.

        Parameters:
        m : int
            Number of rows.
        n : int
            Number of columns.
        pids : list of int
            Process IDs for parallel computation.
        weights : list of float (optional)
            Weights for the processes. Default is an array of ones.
        sbms : list of Future (optional)
            SparseBinMatrix instances. Default is an empty list.
        logic : list of Future (optional)
            ParallelLogic instances. Default is an empty list.
        numblocks : int (optional)
            Number of blocks, default is length(pids) * 2.
        tmp : SharedVector (optional)
            Temporary vector for storing intermediate results.
        sh1 : SharedVector (optional)
            Shared vector of length n for calculations.
        sh2 : SharedVector (optional)
            Shared vector of length n for calculations.
        sems : list of SharedArray (optional)
            List of semaphores for synchronization.
        """
        if weights is None:
            weights = np.ones(len(pids))
        if numblocks is None:
            numblocks = len(pids) * 2

        self.m = m
        self.n = n
        self.pids = pids
        self.weights = weights
        self.sbms = sbms if sbms is not None else []
        self.logic = logic if logic is not None else []
        self.numblocks = numblocks
        self.tmp = tmp
        self.sh1 = sh1
        self.sh2 = sh2
        self.sems = sems

    def size(self):
        return self.m, self.n

    def isempty(self):
        return self.m == 0 or self.n == 0


def intersect(a, b):
    """Returns True if there is an intersection between two arrays."""
    return bool(set(a) & set(b))


def create_parallel_sbm(
    m=None,
    n=None,
    rows=None,
    cols=None,
    pids=None,
    sbms=None,
    logic=None,
    numblocks=None,
    weights=None,
):
    """Creates a ParallelSBM instance from dimensions or from rows and columns.

    If rows and cols are provided, m and n will be inferred from the maximum values in
    those vectors. If m and n are provided, a ParallelSBM will be created with those
    dimensions directly.
    """
    if rows is not None and cols is not None:
        if weights is None:
            weights = np.ones(len(pids))

        if m is None:
            m = int(np.max(rows))
        if n is None:
            n = int(np.max(cols))

        if len(rows) != len(cols):
            raise ValueError("length(rows) must equal length(cols)")

        # Create shared arrays
        shtmp = np.zeros((m,), dtype=float)
        sh1 = np.zeros((n,), dtype=float)
        sh2 = np.zeros((n,), dtype=float)
        sems = make_sems(numblocks, pids)

        # Create the ParallelSBM instance
        ps = ParallelSBM(
            m,
            n,
            pids,
            weights=weights,
            numblocks=numblocks,
            tmp=shtmp,
            sh1=sh1,
            sh2=sh2,
            sems=sems,
        )

        ranges = make_lengths(len(rows), weights)
        mblocks = make_blocks(m, int(numblocks))
        nblocks = make_blocks(n, int(numblocks))

        mblock_grid = np.zeros((numblocks, len(pids)), dtype=int)
        nblock_grid = np.zeros((numblocks, len(pids)), dtype=int)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(len(pids)):
                sbm = SparseBinMatrix(m, n, rows[ranges[i]], cols[ranges[i]])
                sbm_future = executor.submit(
                    lambda sbm=sbm: sbm
                )  # Adjust to your SparseBinMatrix definition
                futures.append(sbm_future)

                # Check for intersections
                for j in range(numblocks):
                    mblock_grid[j, i] = not intersect(sbm.mrange, mblocks[j])
                    nblock_grid[j, i] = not intersect(sbm.nrange, nblocks[j])

            for future in futures:
                sbm_ref = future.result()  # Get the result of SparseBinMatrix
                ps.sbms.append(sbm_ref)

        mblock_counts = np.sum(mblock_grid, axis=1)
        nblock_counts = np.sum(nblock_grid, axis=1)

        for i in range(len(pids)):
            mb = block_order(mblock_counts, np.nonzero(mblock_grid[:, i])[0])
            nb = block_order(nblock_counts, np.nonzero(nblock_grid[:, i])[0])
            pl_future = executor.submit(
                lambda mb=mb, nb=nb: ParallelLogic(
                    mblocks, nblocks, mb, nb, np.zeros(m), np.zeros(n), ps.tmp, ps.sems
                )
            )
            ps.logic.append(pl_future.result())

    else:
        # If m, n, pids, sbms, logic, numblocks are provided directly
        ps = ParallelSBM(m, n, pids, sbms, logic, numblocks)

    return ps


def nonshared(A):
    """Creates a non-shared instance of ParallelSBM."""
    return ParallelSBM(A.m, A.n, A.pids, A.weights, A.sbms, A.logic, A.numblocks)


def make_sems(numblocks, pids):
    """Creates semaphores for synchronization."""
    sems = [Array(np.zeros(16), dtype=np.uint32) for _ in range(numblocks)]
    for sem in sems:
        sem_init(sem)
    return sems


def block_order(counts, blocks):
    """Orders the blocks based on their counts."""
    bcounts = counts[blocks]
    return blocks[np.argsort(bcounts)[::-1]]  # Sort in descending order


def copyto(F, pids):
    """Copies the ParallelSBM instance to new process IDs.

    Parameters:
    F : ParallelSBM
        The original ParallelSBM instance.
    pids : list of int
        New process IDs.

    Returns:
    ParallelSBM
        A new ParallelSBM instance with updated process IDs.
    """
    if len(pids) != len(F.pids):
        raise ValueError(f"length(pids)={len(pids)} must equal length(F.pids)={len(F.pids)}")

    shtmp = Array(np.zeros((F.m,)), dtype=float)
    sh1 = Array(np.zeros((F.n,)), dtype=float)
    sh2 = Array(np.zeros((F.n,)), dtype=float)
    sems = make_sems(F.numblocks, pids)

    ps = ParallelSBM(F.m, F.n, pids, numblocks=F.numblocks, tmp=shtmp, sh1=sh1, sh2=sh2, sems=sems)

    for i in range(len(F.sbms)):
        sbm_ref = F.sbms[i].result()
        ps.sbms.append(sbm_ref)

        logic_ref = F.logic[i].result()
        pl_ref = ParallelLogic(
            logic_ref.mblocks,
            logic_ref.nblocks,
            logic_ref.mblock_order,
            logic_ref.nblock_order,
            np.zeros(F.m),
            np.zeros(F.n),
            ps.tmp,
            ps.sems,
        )
        ps.logic.append(pl_ref)

    return ps


def make_sems(numblocks, pids):
    """Creates semaphores for synchronization."""
    sems = [Array(np.zeros(16, dtype=np.uint32)) for _ in range(numblocks)]
    for sem in sems:
        sem_init(sem)
    return sems


def sem_init(x):
    """Initialize a semaphore."""
    return Semaphore(value=1)


def sem_wait(semaphore):
    """Wait on the semaphore."""
    semaphore.acquire()


def sem_trywait(semaphore):
    """Try to acquire the semaphore without blocking."""
    return semaphore.acquire(block=False)


def sem_post(semaphore):
    """Release the semaphore."""
    semaphore.release()


def gmean(x):
    """Calculate the geometric mean."""
    return np.prod(x) ** (1 / len(x))


def pretty(x):
    """Format the list of floats for pretty printing."""
    return "[" + ", ".join(f"{i:.3f}" for i in x) + "]"


def balanced_parallelsbm(
    rows, cols, pids, numblocks=None, niter=4, ntotal=30, keeplast=4, verbose=False
):
    """Create a balanced parallel SBM instance."""
    if numblocks is None:
        numblocks = len(pids) * 2  # Default number of blocks

    weights = np.ones(len(pids))
    y = Array("d", (int(max(rows)),))  # Shared array for y
    x = Array("d", (int(max(cols)),))  # Shared array for x

    psbm = None  # Placeholder for the ParallelSBM instance

    for i in range(niter):
        psbm = ParallelSBM(rows, cols, pids, numblocks=numblocks, weights=weights)
        times = A_mul_B_time(y, psbm, x, ntotal)

        # Average last few times
        ctimes = np.mean(times[:, -keeplast:], axis=1)
        meantime = gmean(ctimes)

        weights *= (meantime / ctimes) ** (1 / (1 + 0.2 * i))
        weights /= np.sum(weights)  # Normalize weights

        if verbose:
            print(f"{i}. ctimes  = {pretty(ctimes)}")
            print(f"{i}. weights = {pretty(weights)}")

    return psbm


def make_blocks(n, nblocks):
    """Creates block ranges for the specified size and number of blocks.

    Parameters:
    n : int
        Total size.
    nblocks : int
        Number of blocks.

    Returns:
    list of range
        A list of ranges representing blocks.
    """
    bsize = 8 * ((n + nblocks - 1) // nblocks)  # Ceiling of n/nblocks
    if (bsize - 1) * nblocks > n:
        bsize = (n + nblocks - 1) // nblocks
    return [range(1 + (i - 1) * bsize, min(n + 1, i * bsize + 1)) for i in range(1, nblocks + 1)]


def make_lengths(total, weights):
    """Distributes the total length into segments based on weights.

    Parameters:
    total : int
        The total number of elements to distribute.
    weights : array-like
        The weights for each segment.

    Returns:
    list of range
        A list of ranges representing the distributed lengths.
    """
    wnorm = weights / np.sum(weights)  # Normalize weights
    n = np.round(wnorm * total).astype(np.int64)  # Compute the initial distribution
    excess = np.sum(n) - total  # Calculate excess to correct

    i = 0
    while excess != 0:
        n[i] -= np.sign(excess)  # Adjust the current index to balance
        excess = np.sum(n) - total  # Recalculate excess
        i = (i + 1) % len(n)  # Cycle through indices

    ranges = []  # Initialize ranges
    k = 1  # Starting index
    for length in n:
        ranges.append(range(k, k + length))  # Create a range from k to k + length - 1
        k += length  # Update k for the next range

    return ranges


def busywait(x, n, value, ntimes=100):
    """Waits for the value at index n in x to equal the specified value.

    Parameters:
    x : SharedArray
        The shared array to monitor.
    n : int
        The index to check.
    value : int
        The value to wait for.
    ntimes : int, optional
        The number of times to check the value (default is 100).

    Returns:
    bool
        True if the value at x[n] equals the specified value for all checks, otherwise False.
    """
    for _ in range(ntimes):
        if x[n] != value:
            return False
    return True


def A_mul_B(A, x):
    """Performs matrix multiplication with A being either SparseBinMatrix or
    ParallelSBM.

    Parameters:
    A : SparseBinMatrix or ParallelSBM
        The matrix to multiply with.
    x : np.ndarray
        The vector to multiply.

    Returns:
    np.ndarray
        The resulting vector from the multiplication.
    """
    if isinstance(A, SparseBinMatrix):
        return A_mul_B_sparse(A, x)
    elif isinstance(A, ParallelSBM):
        return A_mul_B_parallel(A, x)
    else:
        raise ValueError("Unsupported type for A")


def A_mul_B_sparse(A, x):
    """Computes y = A * x for SparseBinMatrix.

    Parameters:
    A : SparseBinMatrix
        The SparseBinMatrix instance.
    x : np.ndarray
        The input vector to multiply.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    if A.n != len(x):
        raise ValueError(f"A.n={A.n} must equal length(x)={len(x)}")

    y = np.zeros(A.m)
    for i in range(len(A.rows)):
        y[A.rows[i]] += x[A.cols[i]]
    return y


def A_mul_B_parallel(A, x):
    """Computes y = A * x for ParallelSBM.

    Parameters:
    A : ParallelSBM
        The ParallelSBM instance.
    x : np.ndarray
        The input vector to multiply.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    if A.n != len(x):
        raise ValueError(f"A.n={A.n} must equal length(x)={len(x)}")

    y = np.zeros(A.m)
    A.sh1[:] = x
    A_mul_B_shared(A.tmp, A, A.sh1)  # Compute A * x and store in A.tmp
    y[:] = A.tmp  # Assign result to y
    return y


def A_mul_B_shared(y, A, x):
    """Computes y = A * x for ParallelSBM, storing results in a shared array.

    Parameters:
    y : SharedArray
        The output shared array to store results.
    A : ParallelSBM
        The ParallelSBM instance.
    x : SharedArray
        The input shared array to multiply with.
    """
    if A.n != len(x):
        raise ValueError(f"A.n={A.n} must equal length(x)={len(x)}")
    if A.m != len(y):
        raise ValueError(f"A.m={A.m} must equal length(y)={len(y)}")

    y.fill(0)  # Initialize y with zeros
    npids = len(A.pids)

    with ProcessPoolExecutor() as executor:
        futures = []
        for p in range(npids):
            pid = A.pids[p]
            if pid != os.getpid() or npids == 1:
                future = executor.submit(partmul_ref, y, A.sbms[p], A.logic[p], x)
                futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            future.result()  # This will block until the future is done


def At_mul_B(A, x):
    """Computes y = A' * x.

    Parameters:
    A : SparseBinMatrix or ParallelSBM
        The matrix to multiply with.
    x : np.ndarray
        The input vector to multiply.

    Returns:
    np.ndarray
        The resulting vector from the multiplication.
    """
    if isinstance(A, SparseBinMatrix):
        return At_mul_B_sparse(A, x)
    elif isinstance(A, ParallelSBM):
        return At_mul_B_parallel(A, x)
    else:
        raise ValueError("Unsupported type for A")


def At_mul_B_sparse(A, x):
    """Computes y = A' * x for SparseBinMatrix.

    Parameters:
    A : SparseBinMatrix
        The SparseBinMatrix instance.
    x : np.ndarray
        The input vector to multiply.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    if A.m != len(x):
        raise ValueError(f"A.m={A.m} must equal length(x)={len(x)}")

    y = np.zeros(A.n)
    for i in range(len(A.rows)):
        y[A.cols[i]] += x[A.rows[i]]
    return y


def At_mul_B_parallel(A, x):
    """Computes y = A' * x for ParallelSBM and returns the result as a new array.

    Parameters:
    A : ParallelSBM
        The ParallelSBM instance.
    x : np.ndarray
        The input vector to multiply.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    if A.m != len(x):
        raise ValueError(f"A.m={A.m} must equal length(x)={len(x)}")

    A.tmp[:] = x  # Copy x to A.tmp
    At_mul_B_shared(A.sh1, A, A.tmp)  # Perform A' * x and store in A.sh1
    y = np.zeros(A.n)
    y[:] = A.sh1  # Copy results from shared array
    return y


import numpy as np
from multiprocessing import Pool, Array


def partmul_t_ref(y, sbm, logic, x):
    """Placeholder for the transpose multiplication logic."""
    # Implement the actual multiplication logic here.
    # For illustration, we'll add a dummy operation.
    # This function needs to modify the shared array y in place.
    # You might perform something like y += A' * x depending on your implementation.
    pass


def worker_partmul_t_ref(args):
    """Worker function to perform the transpose multiplication."""
    y, sbm, logic, x = args
    partmul_t_ref(y, sbm, logic, x)


def At_mul_B_shared(y, A, x):
    """Computes y = A' * x for ParallelSBM, storing results in a shared array.

    Parameters:
    y : np.ndarray
        The output shared array to store results.
    A : ParallelSBM
        The ParallelSBM instance.
    x : np.ndarray
        The input shared array to multiply with.
    """
    if A.m != len(x):
        raise ValueError(f"A.m={A.m} must equal length(x)={len(x)}")
    if A.n != len(y):
        raise ValueError(f"A.n={A.n} must equal length(y)={len(y)}")

    y.fill(0)  # Reset y to zeros
    npids = len(A.pids)

    # Prepare arguments for each worker
    args = [(y, A.sbms[p], A.logic[p], x) for p in range(npids) if A.pids[p] != os.getpid()]

    # Use multiprocessing Pool to execute partmul_t_ref in parallel
    with Pool(processes=npids) as pool:
        pool.map(worker_partmul_t_ref, args)


def prod_copy(y, v, x):
    """Computes y[i] = v * x[i] for each i in the range of the length of y.

    Parameters:
    y : SharedArray
        The output shared array to store results.
    v : float
        A scalar value to multiply with each element of x.
    x : SharedArray
        The input shared array to multiply.

    Raises:
    ValueError: If the lengths of y and x do not match.
    """
    if len(y) != len(x):
        raise ValueError(f"length(y)={len(y)} must equal length(x)={len(x)}")

    # Perform element-wise multiplication
    for i in range(len(y)):
        y[i] = v * x[i]


def AtA_mul_B(y, A, x, lambd):
    """Computes A'A * x + lambda * x and stores the result in y.

    Parameters:
    y : np.ndarray
        The output array to store results.
    A : ParallelSBM
        The ParallelSBM instance.
    x : np.ndarray
        The input array to multiply with.
    lambd : float
        The regularization parameter.

    Raises:
    ValueError: If the dimensions do not match.
    """
    if A.n != len(x):
        raise ValueError(f"A.n={A.n} must equal length(x)={len(x)}")
    if A.n != len(y):
        raise ValueError(f"A.n={A.n} must equal length(y)={len(y)}")

    tmp = np.zeros_like(x)  # Temporary array to hold results
    npids = len(A.pids)

    # Doing tmp = A * x
    with Pool(processes=npids) as pool:
        # Prepare arguments for the worker function
        args = [
            (tmp, A.sbms[p], A.logic[p], x)
            for p in range(npids)
            if A.pids[p] != os.getpid() or npids == 1
        ]
        pool.map(worker_partmul, args)

    # Doing y += A' * tmp
    with Pool(processes=npids) as pool:
        # Prepare arguments for the worker function
        args = [
            (y, A.sbms[p], A.logic[p], tmp)
            for p in range(npids)
            if A.pids[p] != os.getpid() or npids == 1
        ]
        pool.map(worker_partmul_t, args)

    # Apply the regularization term: y += lambda * x
    y += lambd * x


def worker_partmul(args):
    """Worker function for partmul_ref."""
    tmp, sbm, logic, x = args
    partmul_ref(tmp, sbm, logic, x)


def worker_partmul_t(args):
    """Worker function for partmul_t_ref."""
    y, sbm, logic, tmp = args
    partmul_t_ref(y, sbm, logic, tmp)


def worker_partmul_time(args):
    """Worker function to perform partmul_time in a process."""
    y, sbm, logic, x = args
    return partmul_time(y, sbm, logic, x)


def A_mul_B_time(y, A, x, ntimes):
    """Computes the multiplication of A and x multiple times and returns the timing.

    Parameters:
    y : np.ndarray
        The output shared array to store results.
    A : ParallelSBM
        The ParallelSBM instance.
    x : np.ndarray
        The input shared array to multiply with.
    ntimes : int
        The number of times to perform the multiplication.

    Raises:
    ValueError: If the dimensions do not match.

    Returns:
    np.ndarray
        An array of times for each process and each iteration.
    """
    if A.n != len(x):
        raise ValueError(f"A.n={A.n} must equal length(x)={len(x)}")
    if A.m != len(y):
        raise ValueError(f"A.m={A.m} must equal length(y)={len(y)}")

    ptime = np.zeros((len(A.pids), ntimes))

    for i in range(ntimes):
        y.fill(0)  # Clear the output array
        # Prepare arguments for each process
        args = []
        for p in range(len(A.pids)):
            pid = A.pids[p]
            if pid != os.getpid() or len(A.pids) == 1:
                # Use multiprocessing to handle remote calls
                args.append((y, A.sbms[p], A.logic[p], x))

        # Use multiprocessing Pool to execute partmul_time
        with Pool(processes=len(A.pids)) as pool:
            results = pool.map(worker_partmul_time, args)

            # Store the times returned by each worker
            for p, result in enumerate(results):
                if p < len(A.pids):
                    ptime[p, i] = result

    return ptime


def copy(to, from_, rng):
    """Copies elements from the 'from_' array to the 'to' array over the specified
    range.

    Parameters:
    to : np.ndarray
        The destination array where elements will be copied.
    from_ : np.ndarray
        The source array from which elements will be copied.
    rng : range
        The range of indices to copy.

    Returns:
    None
    """
    to[rng] = from_[rng]


def add_elements(to, from_, rng):
    """Adds elements from the 'from_' array to the 'to' array over the specified range.

    Parameters:
    to : np.ndarray
        The destination array to which elements will be added.
    from_ : np.ndarray
        The source array from which elements will be added.
    rng : range
        The range of indices to add.

    Returns:
    None
    """
    to[rng] += from_[rng]


def add_over_range(y, x, range_x, yrange):
    """Adds elements from array x to array y over specified ranges.

    Parameters:
    y : np.ndarray
        The destination array to which elements will be added.
    x : np.ndarray
        The source array from which elements will be added.
    range_x : range
        The range of indices from array x to add.
    yrange : range
        The range of indices in array y to which the elements will be added.

    Returns:
    None
    """
    y[yrange] += x[range_x]  # Ensure ranges are correct


def rangefill(x, v, rng):
    """Fills the specified range of the array with the value 'v'.

    Parameters:
    x : np.ndarray
        The array to fill.
    v : scalar
        The value to fill the array with.
    rng : range
        The range of indices to fill.

    Returns:
    None
    """
    x[rng] = v


def partmul_time(y, Aref, logicref, x):
    """Performs a parallel multiplication and times the operation.

    Parameters:
    y : SharedArray
        The output shared array to store results.
    Aref : Future
        A reference to a SparseBinMatrix.
    logicref : Future
        A reference to a ParallelLogic instance.
    x : SharedArray
        The input shared array to multiply with.

    Returns:
    float
        The time taken for the multiplication.
    """
    # Replace fetch with Future.result()
    A = Aref.result()  # Retrieve the SparseBinMatrix
    logic = logicref.result()  # Retrieve the ParallelLogic instance
    start_time = time.time()  # Start the timer
    partmul(y, A, logic, x)  # Perform the multiplication
    return time.time() - start_time  # Return elapsed time


def partmul_ref(y, Aref, logicref, x):
    """Performs a multiplication with references for SparseBinMatrix and ParallelLogic.

    Parameters:
    y : SharedArray
        The output shared array to store results.
    Aref : Future
        A reference to a SparseBinMatrix.
    logicref : Future
        A reference to a ParallelLogic instance.
    x : SharedArray
        The input shared array to multiply with.

    Returns:
    None
    """
    # Replace fetch with Future.result()
    A = Aref.result()  # Retrieve the SparseBinMatrix
    logic = logicref.result()  # Retrieve the ParallelLogic instance
    partmul(y, A, logic, x)  # Perform the multiplication


def partmul_t_ref(y, Aref, logicref, x):
    """Performs a transpose multiplication with references for SparseBinMatrix and
    ParallelLogic.

    Parameters:
    y : SharedArray
        The output shared array to store results.
    Aref : Future
        A reference to a SparseBinMatrix.
    logicref : Future
        A reference to a ParallelLogic instance.
    x : SharedArray
        The input shared array to multiply with.

    Returns:
    None
    """
    # Replace fetch with Future.result()
    A = Aref.result()  # Retrieve the SparseBinMatrix
    logic = logicref.result()  # Retrieve the ParallelLogic instance
    partmul_t(y, A, logic, x)  # Perform the transpose multiplication


def range_fill(array, value, rng):
    """Fills the given range of the array with the specified value."""
    array[rng] = value


def partmul(y, A, logic, x):
    """Performs the matrix multiplication y = A * x.

    Parameters:
    y : numpy.ndarray
        The output shared array.
    A : SparseBinMatrix
        The sparse binary matrix.
    logic : ParallelLogic
        The logic for local arrays and synchronization.
    x : numpy.ndarray
        The input shared array.

    Returns:
    None
    """
    ylocal = logic.localm
    xlocal = logic.localn
    range_fill(ylocal, 0, A.mrange)  # Fill with zero
    np.copyto(xlocal, x[A.nrange])  # Copy x into xlocal for specified range

    rows = A.rows
    cols = A.cols

    for i in range(len(rows)):
        ylocal[rows[i]] += xlocal[cols[i]]

    # Add the result to the shared array
    addshared(y, ylocal, logic.sems, logic.mblocks, logic.mblock_order, A.mrange)


def partmul_t(y, A, logic, x):
    """Performs the transpose matrix multiplication y = A' * x.

    Parameters:
    y : numpy.ndarray
        The output shared array.
    A : SparseBinMatrix
        The sparse binary matrix.
    logic : ParallelLogic
        The logic for local arrays and synchronization.
    x : numpy.ndarray
        The input shared array.

    Returns:
    None
    """
    ylocal = logic.localn
    xlocal = logic.localm
    range_fill(ylocal, 0, A.nrange)  # Fill with zero
    np.copyto(xlocal, x[A.mrange])  # Copy x into xlocal for specified range

    rows = A.cols
    cols = A.rows

    for i in range(len(rows)):
        ylocal[rows[i]] += xlocal[cols[i]]

    # Add the result to the shared array
    addshared(y, ylocal, logic.sems, logic.nblocks, logic.nblock_order, A.nrange)


def addshared(y, x, sems, ranges, order, yrange):
    """Adds x to y using shared locking mechanisms.

    Parameters:
    y : numpy.ndarray
        The output shared array.
    x : numpy.ndarray
        The input shared array to add.
    sems : list
        List of semaphore objects for synchronization.
    ranges : list
        The ranges for each block.
    order : list
        The order of blocks.
    yrange : range
        The range to operate on in y.

    Returns:
    None
    """
    blocks = order.copy()
    nblocks = len(blocks)

    while True:
        i = next((idx for idx, val in enumerate(blocks) if val > 0), -1)
        if i == -1:  # No more blocks
            return

        for j in range(i, nblocks):
            block = blocks[j]
            if block == 0:
                continue

            # Try to acquire the lock
            if sems[block].acquire(blocking=False):  # Try to get the lock
                # Copy the result to the shared array
                add_over_range(y, x, ranges[blocks[j]], yrange)
                sems[block].release()  # Release the lock
                blocks[j] = 0  # Mark this block as done
                break


def worker_cg_AtA_ref(args):
    """Worker function for processing the conjugate gradient computation."""
    return cg_AtA_ref(*args)


def solve_cg2(Frefs, rhs, lambda_, tol=1e-6, maxiter=None):
    """Solves the linear system using the conjugate gradient method with parallel
    operations on Future references.

    Parameters:
    Frefs : list
        A list of Future references (in Python, this would be callable objects).
    rhs : np.ndarray
        The right-hand side matrix (2D).
    lambda_ : float
        The regularization parameter.
    tol : float, optional
        The tolerance for convergence.
    maxiter : int, optional
        The maximum number of iterations.

    Returns:
    np.ndarray
        The computed beta values.
    """
    beta = np.zeros(rhs.shape)  # Initialize beta with the same shape as rhs
    D = rhs.shape[1]

    maxiter = maxiter if maxiter is not None else rhs.shape[0]

    # Prepare arguments for the worker
    args = [(Frefs[idx], rhs[:, idx], lambda_, tol, maxiter) for idx in range(D)]

    # Use multiprocessing Pool to execute tasks in parallel
    with Pool(processes=None) as pool:  # None uses all available CPU cores
        results = pool.map(worker_cg_AtA_ref, args)

    # Store the results in beta
    for idx, result in enumerate(results):
        beta[:, idx] = result

    return beta


def A_mul_B_ref(Fref, x):
    """Multiplies the matrix associated with the Future reference Fref by the vector x.

    Parameters:
    Fref : Future
        A Future reference that resolves to a matrix.
    x : np.ndarray
        The vector to multiply with.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    F = Fref.result()
    return np.dot(F, x)


def At_mul_B_ref(Fref, x):
    """Multiplies the transpose of the matrix associated with the Future reference Fref
    by the vector x.

    Parameters:
    Fref : Future
        A Future reference that resolves to a matrix.
    x : np.ndarray
        The vector to multiply with.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    F = Fref.result()
    return At_mul_B(F, x)


def Frefs_mul_B(Frefs, x):
    """Computes y = F * x in parallel (along columns of x).

    Parameters:
    Frefs : list
        A list of Future references.
    x : np.ndarray
        The input matrix.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    m, n = Frefs[0].result().shape
    if n != x.shape[0]:
        raise ValueError(f"Frefs.n={n} must equal length(x)={x.shape[0]}")

    y = np.zeros((m, x.shape[1]))  # Initialize the output matrix
    D = x.shape[1]
    i = 1

    def nextidx():
        nonlocal i
        idx = i
        i += 1
        return idx

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(A_mul_B_ref, ref, x[:, idx]): idx
            for idx, ref in enumerate(Frefs)
            if idx < D
        }

        for future in as_completed(futures):
            idx = futures[future]
            y[:, idx] = future.result()  # Store the result in y

    return y


def Frefs_t_mul_B(Frefs, x):
    """Computes y = F' * x in parallel (along columns of x).

    Parameters:
    Frefs : list
        A list of Future references.
    x : np.ndarray
        The input matrix.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    m, n = Frefs[0].result().shape
    if m != x.shape[0]:
        raise ValueError(f"Frefs.m={m} must equal length(x)={x.shape[0]}")

    y = np.zeros((n, x.shape[1]))  # Initialize the output matrix
    D = x.shape[1]
    i = 1

    def nextidx():
        nonlocal i
        idx = i
        i += 1
        return idx

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(At_mul_B_ref, ref, x[:, idx]): idx
            for idx, ref in enumerate(Frefs)
            if idx < D
        }

        for future in as_completed(futures):
            idx = futures[future]
            y[:, idx] = future.result()  # Store the result in y

    return y


def xy2d(n, x, y):
    """Converts (x, y) coordinates to a Hilbert order index d.

    Parameters:
    n : int
        The dimension size, should be a power of 2.
    x : int
        The x coordinate.
    y : int
        The y coordinate.

    Returns:
    int
        The Hilbert order index.
    """
    d = 0
    s = n // 2  # Equivalent to div(n, 2)

    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)  # XOR in Python
        x, y = rot(s, x, y, rx, ry)
        s //= 2  # Equivalent to div(s, 2)

    return d


def rot(n, x, y, rx, ry):
    """Rotates the (x, y) coordinates based on the rx and ry values.

    Parameters:
    n : int
        The size of the grid.
    x : int
        The x coordinate.
    y : int
        The y coordinate.
    rx : bool
        The rotation flag for x.
    ry : bool
        The rotation flag for y.

    Returns:
    tuple
        The rotated (x, y) coordinates.
    """
    if not ry:  # Equivalent to ry == false
        if rx:  # Equivalent to rx == true
            x = n - 1 - x
            y = n - 1 - y
        return y, x  # Return as (y, x)
    return x, y  # Return as (x, y)


def sort_hilbert(rows, cols):
    """Sorts the rows and columns according to their Hilbert order.

    Parameters:
    rows : np.ndarray
        The array of row indices.
    cols : np.ndarray
        The array of column indices.

    Returns:
    tuple
        Sorted row and column indices.
    """
    maxrc = max(np.max(rows), np.max(cols))
    n = 2 ** round(np.ceil(np.log2(maxrc)))  # Equivalent to 2 ^ round(Int, ceil(log2(maxrc)))

    h = np.zeros(len(rows), dtype=int)

    for i in range(len(h)):
        h[i] = xy2d(n, rows[i] - 1, cols[i] - 1)  # Adjust indices to be 0-based

    hsorted = np.argsort(h)  # Get indices that would sort h

    return rows[hsorted], cols[hsorted]
