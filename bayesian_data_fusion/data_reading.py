import numpy as np
from scipy.sparse import coo_matrix


def read_ecfp(filename):
    """Reads an ECFP file and extracts fingerprint information.

    Parameters:
    filename: str, Path to the ECFP file.

    Returns:
    tuple: Contains rows, cols, and fp:
        - rows: List of row indices.
        - cols: List of column indices (fingerprint IDs).
        - fp: Dictionary mapping raw fingerprints to unique IDs.
    """
    i = 0  # Line counter
    next_fp = 1  # Next available fingerprint ID
    fp = {}  # Dictionary to map raw fingerprints to unique IDs
    rows = []  # List of row indices
    cols = []  # List of column indices (fingerprint IDs)

    # Open and read the file
    with open(filename, "r") as f:
        for line in f:
            i += 1
            a = line.strip().split(",")
            for j in range(1, len(a)):
                fp_raw = int(a[j])  # Parse the fingerprint as an integer

                # Fetch or assign a unique fingerprint ID
                if fp_raw in fp:
                    fp_id = fp[fp_raw]
                else:
                    fp_id = next_fp
                    fp[fp_raw] = fp_id
                    next_fp += 1

                # Append to rows and cols
                rows.append(i)
                cols.append(fp_id)

    print(f"Number of lines: {i}")
    return rows, cols, fp


def read_rowcol(filename):
    """Reads row and column indices from a file.

    Parameters:
    filename: str, Path to the file containing row and column data.

    Returns:
    tuple: Contains two lists:
        - rows: List of row indices.
        - cols: List of column indices.
    """
    rows = []
    cols = []

    with open(filename, "r") as f:
        for line in f:
            a = line.strip().split(",")
            rows.append(int(a[0]))  # Parse the first column as row index
            cols.append(int(a[1]))  # Parse the second column as column index

    return rows, cols


def read_binary_int32(filename):
    """Reads a binary file containing an Int64 header (number of rows and columns)
    followed by a matrix of Int32 values.

    Parameters:
    filename: str, Path to the binary file.

    Returns:
    np.array: A 2D NumPy array of Int32 values.
    """
    with open(filename, "rb") as f:
        # Read number of rows and columns as Int64
        nrows = int(np.fromfile(f, dtype=np.int64, count=1))
        ncols = int(np.fromfile(f, dtype=np.int64, count=1))
        # Read the remaining data as a matrix of Int32
        return np.fromfile(f, dtype=np.int32).reshape((nrows, ncols))


def read_binary_float32(filename):
    """Reads a binary file containing an Int64 header (number of rows and columns)
    followed by a matrix of Float32 values.

    Parameters:
    filename: str, Path to the binary file.

    Returns:
    np.array: A 2D NumPy array of Float32 values.
    """
    with open(filename, "rb") as f:
        # Read number of rows and columns as Int64
        nrows = int(np.fromfile(f, dtype=np.int64, count=1))
        ncols = int(np.fromfile(f, dtype=np.int64, count=1))
        # Read the remaining data as a matrix of Float32
        return np.fromfile(f, dtype=np.float32).reshape((nrows, ncols))


def read_sparse_float32(filename):
    """Reads a binary file representing a sparse matrix in COO format with Int32 rows
    and columns and Float32 values.

    Parameters:
    filename: str, Path to the binary file.

    Returns:
    tuple: Contains three arrays:
        - rows: Array of row indices (Int32).
        - cols: Array of column indices (Int32).
        - vals: Array of values (Float32).
    """
    with open(filename, "rb") as f:
        # Read the number of non-zero entries (nnz)
        nnz = int(np.fromfile(f, dtype=np.int64, count=1))

        # Read rows, columns, and values
        rows = np.fromfile(f, dtype=np.int32, count=nnz)
        cols = np.fromfile(f, dtype=np.int32, count=nnz)
        vals = np.fromfile(f, dtype=np.float32, count=nnz)

        return rows, cols, vals


def read_sparse(filename):
    """Reads a file with row and column indices and creates a sparse matrix in COO
    format.

    Parameters:
    filename: str, Path to the file with row and column data.

    Returns:
    scipy.sparse.coo_matrix: A sparse matrix with ones as values at specified row and column indices.
    """
    # Assume read_rowcol is defined elsewhere, and it reads row and column indices from a file
    rows, cols = read_rowcol(filename)

    # Create a sparse matrix with a value of 1.0 (float) at each (row, col) index
    return coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)))


def filter_rare(X, nmin):
    """Filters out columns from a sparse matrix where the column sum is less than nmin.

    Parameters:
    X: csc_matrix, A sparse matrix in CSC format.
    nmin: int, Minimum sum threshold for keeping a column.

    Returns:
    csc_matrix: A filtered sparse matrix with columns having sums >= nmin.
    """
    # Calculate column sums and filter columns based on nmin
    featn = np.array(X.sum(axis=0)).flatten()
    return X[:, featn >= nmin]


def write_binary_int32(filename, X):
    """Writes an Int32 matrix to a binary file.

    Parameters:
    filename: str, Path to the binary file.
    X: np.array, A matrix of Int32 values.
    """
    write_binary_matrix(filename, X)


def write_binary_matrix(filename, X):
    """Writes a matrix to a binary file with dimensions (Int64) followed by the matrix
    data.

    Parameters:
    filename: str, Path to the binary file.
    X: np.array, A 2D matrix to write to the file.
    """
    with open(filename, "wb") as f:
        # Write dimensions as Int64
        np.array([X.shape[0], X.shape[1]], dtype=np.int64).tofile(f)
        # Write the matrix data
        X.tofile(f)


def write_sparse_float32(filename, X):
    """Writes a sparse CSC matrix to a binary file in COO format with Int32 rows and
    columns and Float32 values.

    Parameters:
    filename: str, Path to the binary file.
    X: csc_matrix, A sparse matrix in CSC format.
    """
    # Get non-zero entries in COO format
    I, J, V = X.nonzero()  # I: row indices, J: column indices
    values = X.data  # Non-zero values

    # Convert to appropriate types
    rows = I.astype(np.int32)
    cols = J.astype(np.int32)
    values = values.astype(np.float32)

    # Call the helper function to write to file
    write_sparse_float32_helper(filename, rows, cols, values)


def write_sparse_float32_helper(filename, rows, cols, values):
    """Writes row indices, column indices, and values of a sparse matrix to a binary
    file.

    Parameters:
    filename: str, Path to the binary file.
    rows: np.array of Int32, Row indices of non-zero entries.
    cols: np.array of Int32, Column indices of non-zero entries.
    values: np.array of Float32, Values of non-zero entries.
    """
    with open(filename, "wb") as f:
        # Write the number of non-zero elements as Int64
        np.array([len(rows)], dtype=np.int64).tofile(f)

        # Write the row indices, column indices, and values
        rows.tofile(f)
        cols.tofile(f)
        values.tofile(f)


def write_sparse_binary_matrix(filename, X):
    """Writes the non-zero coordinates of a sparse CSC matrix to a binary file.

    Parameters:
    filename: str, Path to the binary file.
    X: csc_matrix, A sparse matrix in CSC format.
    """
    # Get the non-zero row indices, column indices, and values
    rows, cols = X.nonzero()  # row and column indices of non-zero entries

    with open(filename, "wb") as f:
        # Write matrix dimensions (nrows, ncols) and number of non-zero entries (nnz) as Int64
        np.array([X.shape[0], X.shape[1], X.nnz], dtype=np.int64).tofile(f)

        # Write row and column indices as Int32
        rows.astype(np.int32).tofile(f)
        cols.astype(np.int32).tofile(f)


def read_sparse_binary_matrix(filename):
    """Reads a sparse matrix from a binary file written in COO format with specified
    rows and columns.

    Parameters:
    filename: str, Path to the binary file.

    Returns:
    csc_matrix: A sparse matrix reconstructed from the binary file.
    """
    with open(filename, "rb") as f:
        # Read matrix dimensions and number of non-zero elements (nnz) as Int64
        nrows = int(np.fromfile(f, dtype=np.int64, count=1))
        ncols = int(np.fromfile(f, dtype=np.int64, count=1))
        nnz = int(np.fromfile(f, dtype=np.int64, count=1))

        # Read row and column indices as Int32
        rows = np.fromfile(f, dtype=np.int32, count=nnz)
        cols = np.fromfile(f, dtype=np.int32, count=nnz)

        # Reconstruct the sparse matrix with ones at specified coordinates
        return coo_matrix(
            (np.ones(nnz, dtype=np.float32), (rows, cols)), shape=(nrows, ncols)
        ).tocsc()


def read_matrix_market(filename):
    """Reads a sparse matrix from a Matrix Market file.

    Parameters:
    filename: str, Path to the Matrix Market file.

    Returns:
    coo_matrix: A sparse matrix in COO format.
    """
    nrows, ncols, nnz = 0, 0, 0
    rows = []
    cols = []
    vals = []

    with open(filename, "r") as f:
        # Reading the first non-comment line to get dimensions and nnz
        while True:
            line = f.readline().strip()
            if not line.startswith("%"):
                arr = line.split()
                nrows = int(arr[0])
                ncols = int(arr[1])
                nnz = int(arr[2])
                break

        # Initialize arrays for rows, columns, and values
        rows = np.zeros(nnz, dtype=np.int32)
        cols = np.zeros(nnz, dtype=np.int32)
        vals = np.zeros(nnz, dtype=np.float64)

        # Read the remaining lines
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                continue
            arr = line.split()
            rows[i] = int(arr[0]) - 1  # Convert to 0-based index
            cols[i] = int(arr[1]) - 1  # Convert to 0-based index
            vals[i] = float(arr[2])
            i += 1

    # Construct and return the sparse matrix
    return coo_matrix((vals, (rows, cols)), shape=(nrows, ncols))


def write_matrix_market(filename, X):
    """Writes a DataFrame to a file in Matrix Market format.

    Parameters:
    filename: str, Path to the output file.
    X: pd.DataFrame, DataFrame with three columns representing row indices, column indices, and values.
    """
    # Determine matrix dimensions and number of non-zero elements
    nrows = X.iloc[:, 0].max()
    ncols = X.iloc[:, 1].max()
    nnz = X.shape[0]

    # Open the file and write the header line
    with open(filename, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{nrows}\t{ncols}\t{nnz}\n")

        # Write the matrix data (assumes X has three columns: row, column, value)
        np.savetxt(f, X.iloc[:, :3].values, fmt="%d\t%d\t%.6f")


def write_sparse_float64(filename, X):
    """Writes a sparse matrix in CSC format to a binary file with Float64 values.

    Parameters:
    filename: str, Path to the binary file.
    X: csc_matrix, A sparse matrix in CSC format.
    """
    # Get non-zero row indices, column indices, and values
    rows, cols = X.nonzero()  # row and column indices of non-zero entries
    values = X.data  # Non-zero values

    with open(filename, "wb") as f:
        # Write the dimensions (number of rows and columns) and number of non-zero entries as Int64
        np.array([X.shape[0], X.shape[1], X.nnz], dtype=np.int64).tofile(f)

        # Write row indices, column indices as Int32, and values as Float64
        rows.astype(np.int32).tofile(f)
        cols.astype(np.int32).tofile(f)
        values.astype(np.float64).tofile(f)


def read_sparse_float64(filename):
    """Reads a sparse matrix from a binary file written with Float64 values.

    Parameters:
    filename: str, Path to the binary file.

    Returns:
    csc_matrix: A sparse matrix reconstructed from the binary file.
    """
    with open(filename, "rb") as f:
        # Read the matrix dimensions and number of non-zero entries as Int64
        nrow = int(np.fromfile(f, dtype=np.int64, count=1))
        ncol = int(np.fromfile(f, dtype=np.int64, count=1))
        nnz = int(np.fromfile(f, dtype=np.int64, count=1))

        # Read row indices, column indices as Int32, and values as Float64
        rows = np.fromfile(f, dtype=np.int32, count=nnz)
        cols = np.fromfile(f, dtype=np.int32, count=nnz)
        values = np.fromfile(f, dtype=np.float64, count=nnz)

        # Construct and return the sparse matrix
        return coo_matrix((values, (rows, cols)), shape=(nrow, ncol)).tocsc()
