class IndexedDF:
    def __init__(self, df, dims=None):
        """Initializes an IndexedDF with a DataFrame and a multidimensional index.

        Parameters:
        df: pd.DataFrame, The DataFrame to index.
        dims: list of int or tuple, Dimensions for indexing, based on maximum values in each column if not provided.
        """
        self.df = df

        # Determine dimensions if not provided
        if dims is None:
            dims = [df.iloc[:, i].max() for i in range(df.shape[1] - 1)]
        elif isinstance(dims, tuple):
            dims = [int(d) for d in dims]

        # Initialize index with nested lists
        self.index = [[[] for _ in range(dim + 1)] for dim in dims]

        # Populate the index with row indices for each dimension
        for i in range(df.shape[0]):  # Iterate over rows
            for mode in range(len(dims)):
                j = df.iloc[i, mode]
                self.index[mode][j].append(i)

    @classmethod
    def from_tuple(cls, df, dims):
        """Alternative constructor that accepts dimensions as a tuple."""
        return cls(df, dims=tuple(dims))

    @classmethod
    def default_index(cls, df):
        """Alternative constructor that infers dimensions from DataFrame columns."""
        dims = [df.iloc[:, i].max() for i in range(df.shape[1] - 1)]
        return cls(df, dims=dims)


def value_mean(idf):
    """Calculates the mean of the last column in the IndexedDF's DataFrame.

    Parameters:
    idf: IndexedDF, The IndexedDF object.

    Returns:
    float: The mean value of the last column in idf.df.
    """
    return idf.df.iloc[:, -1].mean()


def size(idf, dim=None):
    """Returns the size of the IndexedDF object in terms of the dimensions of the index.

    Parameters:
    idf: IndexedDF, The IndexedDF object.
    dim: int, Optional, specific dimension to get the size for.

    Returns:
    tuple or int: The sizes of each dimension as a tuple if dim is None, otherwise the size of the specified dimension.
    """
    if dim is None:
        return tuple(len(i) for i in idf.index)
    else:
        return len(idf.index[dim])


def nnz(idf):
    """Returns the number of non-zero elements in the IndexedDF, defined as the number
    of rows in the DataFrame.

    Parameters:
    idf: IndexedDF, The IndexedDF object.

    Returns:
    int: The number of rows in idf.df.
    """
    return idf.df.shape[0]


def remove_samples(idf, samples):
    """Removes specified samples (rows) from the IndexedDF's DataFrame and returns a new
    IndexedDF.

    Parameters:
    idf: IndexedDF, The IndexedDF object.
    samples: list of int, The row indices to remove.

    Returns:
    IndexedDF: A new IndexedDF with specified samples removed.
    """
    remaining_rows = idf.df.drop(samples)
    return IndexedDF(remaining_rows.reset_index(drop=True), size(idf))


def get_values(idf):
    """Returns the values of the last column in the IndexedDF's DataFrame as an array.

    Parameters:
    idf: IndexedDF, The IndexedDF object.

    Returns:
    np.array: Array of values from the last column in idf.df.
    """
    return idf.df.iloc[:, -1].to_numpy()


def get_mode(idf, mode):
    """Returns the values of the specified mode (column) in the IndexedDF's DataFrame.

    Parameters:
    idf: IndexedDF, The IndexedDF object.
    mode: int, Column index to retrieve.

    Returns:
    pd.Series: The values of the specified column.
    """
    return idf.df.iloc[:, mode]


def get_data(idf, mode, i):
    """Returns the rows in the IndexedDF's DataFrame that correspond to the given index
    in the specified mode.

    Parameters:
    idf: IndexedDF, The IndexedDF object.
    mode: int, The dimension or column mode.
    i: int, The specific index within the mode to retrieve data for.

    Returns:
    pd.DataFrame: Rows corresponding to the index in the specified mode.
    """
    return idf.df.iloc[idf.index[mode][i]]


def get_count(idf, mode, i):
    """Returns the count of rows associated with the given index in the specified mode.

    Parameters:
    idf: IndexedDF, The IndexedDF object.
    mode: int, The dimension or column mode.
    i: int, The specific index within the mode.

    Returns:
    int: The count of rows for the specified index in the given mode.
    """
    return len(idf.index[mode][i])


def get_i(idf, mode, i):
    """Returns the row indices associated with the given index in the specified mode.

    Parameters:
    idf: IndexedDF, The IndexedDF object.
    mode: int, The dimension or column mode.
    i: int, The specific index within the mode.

    Returns:
    list of int: Row indices associated with the specified index in the given mode.
    """
    return idf.index[mode][i]


class FastIDF:
    def __init__(self, ids, values, index):
        """Initializes a FastIDF object, used in sampling of latent variables.

        Parameters:
        ids: np.array, 2D array representing IDs.
        values: np.array, Array of values associated with the IDs.
        index: list of lists of lists, Nested index structure for efficient sampling.
        """
        self.ids = ids  # 2D matrix of IDs
        self.values = values  # Array of values
        self.index = index  # Nested list index structure

    @classmethod
    def from_indexed_df(cls, idf):
        """Alternative constructor to create FastIDF from an IndexedDF object.

        Parameters:
        idf: IndexedDF, The IndexedDF object.

        Returns:
        FastIDF: A new FastIDF instance.
        """
        ids = idf.df.iloc[:, :-1].to_numpy()  # Convert all columns except last to an array
        values = idf.df.iloc[:, -1].to_numpy()  # Convert the last column to an array
        return cls(ids, values, idf.index)

    @classmethod
    def from_dataframe(cls, df, dims):
        """Alternative constructor to create FastIDF from a DataFrame and specified
        dimensions.

        Parameters:
        df: pd.DataFrame, The DataFrame to index.
        dims: list of int, Dimensions for indexing.

        Returns:
        FastIDF: A new FastIDF instance.
        """
        # Initialize nested index structure based on dims
        index = [[[] for _ in range(dim + 1)] for dim in dims]

        # Populate the index with row indices for each dimension
        for i in range(df.shape[0]):  # Iterate over rows
            for mode in range(len(dims)):
                j = df.iloc[i, mode]
                index[mode][j].append(i)

        ids = df.iloc[:, :-1].to_numpy()  # Convert all columns except last to an array
        values = df.iloc[:, -1].to_numpy()  # Convert the last column to an array

        return cls(ids, values, index)

    def get_data(self, mode, i):
        """Retrieves the IDs and values associated with a specific mode and index.

        Parameters:
        mode: int, The mode (dimension) to retrieve data from.
        i: int, The specific index within the mode.

        Returns:
        tuple: A tuple containing the IDs and values for the specified index.
        """
        id_list = self.index[mode][i]
        return self.ids[id_list, :], self.values[id_list]

    def size(self, dim=None):
        """Returns the size of the FastIDF object in terms of the dimensions of the
        index.

        Parameters:
        dim: int, Optional, specific dimension to get the size for.

        Returns:
        tuple or int: The sizes of each dimension as a tuple if dim is None,
        otherwise the size of the specified dimension.
        """
        if dim is None:
            return tuple(len(i) for i in self.index)
        else:
            return len(self.index[dim])

    def nnz(self):
        """Returns the number of non-zero elements in the FastIDF, defined as the length
        of the values array.

        Returns:
        int: The number of values (non-zero elements).
        """
        return len(self.values)
