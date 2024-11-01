import numpy as np
from scipy.sparse import coo_matrix
from time import time


def macau_sgd(
    data,
    num_latent=10,
    verbose=True,
    niter=100,
    lrate=0.001,
    reset_model=True,
    batch_size=4000,
    clamp=None,
):
    """Performs Macau's Stochastic Gradient Descent (SGD) algorithm for a relation data
    model.

    Parameters:
    data : RelationData, The dataset to model.
    num_latent : int, Number of latent factors.
    verbose : bool, If True, print progress and information.
    niter : int, Number of SGD iterations.
    lrate : float, Learning rate.
    reset_model : bool, If True, reset the model before training.
    batch_size : int, Number of samples in each minibatch.
    clamp : list, Optional clamp for predicted values.

    Returns:
    tuple: Ubatches and Vbatches (the minibatches for U and V samples).
    """
    if clamp is None:
        clamp = []

    # Initialization
    if verbose:
        print("Model setup")
    if reset_model:
        reset(data, num_latent)

    rmse = np.nan
    rmse_sample = np.nan
    rmse_train = np.nan

    # Data preparation
    df = data.relations[0].data.df
    mean_value = df.iloc[:, -1].mean()
    uid = df.iloc[:, 0].astype(np.int32).values
    vid = df.iloc[:, 1].astype(np.int32).values
    val = df.iloc[:, 2].values - mean_value

    Ubatches = create_minibatches(vid, uid, val, batch_size)
    Vbatches = [batch.T for batch in Ubatches]

    Usample = data.entities[0].model.sample
    Vsample = data.entities[1].model.sample

    test_uid = data.relations[0].test_vec.iloc[:, 0].astype(int).values
    test_vid = data.relations[0].test_vec.iloc[:, 1].astype(int).values
    test_val = data.relations[0].test_vec.iloc[:, 2].values
    test_idx = np.column_stack((test_uid, test_vid))

    alpha = data.relations[0].model.alpha
    yhat_post = np.zeros(len(test_val))

    # SGD iterations over mini-batches
    start_time = time()
    for i in range(len(Ubatches)):
        sgd_update(Usample, Vsample, Ubatches[i], data.entities[0].model, alpha, lrate)
        sgd_update(Vsample, Usample, Vbatches[i], data.entities[1].model, alpha, lrate)
    end_time = time()

    if verbose:
        print(f"SGD completed in {end_time - start_time:.2f} seconds.")

    return Ubatches, Vbatches


def create_minibatches(uid, vid, val, bsize):
    """Creates minibatches from the given user and item indices and values.

    Parameters:
    uid : np.array, Array of user indices.
    vid : np.array, Array of item indices.
    val : np.array, Array of values associated with (user, item) pairs.
    bsize : int, Batch size for each minibatch.

    Returns:
    list : List of sparse matrices in CSR format, each representing a minibatch.
    """
    perm = np.random.permutation(len(uid))
    batches = []
    umax = uid.max()
    vmax = vid.max()

    for i in range(0, len(uid), bsize):
        j = min(len(uid), i + bsize)
        batch = coo_matrix(
            (val[perm[i:j]], (uid[perm[i:j]] - 1, vid[perm[i:j]] - 1)),  # Adjusted for 0-indexing
            shape=(umax, vmax),
        ).tocsr()
        batches.append(batch)

    return batches


def sgd_update(sample, Vsample, batch, model, alpha):
    """Performs an in-place stochastic gradient descent (SGD) update on the latent
    matrix `sample`.

    Parameters:
    sample : np.array, The matrix to update in place.
    Vsample : np.array, The matrix representing the other entity in the relation.
    batch : scipy.sparse.csc_matrix, A sparse matrix batch in CSC format.
    model : EntityModel, Contains model parameters like Lambda and mu.
    alpha : float, Regularization parameter.
    """
    for n in range(sample.shape[1]):
        grad(n, sample, Vsample, batch, model.Lambda, model.mu, alpha)


def grad(n, sample, Vsample, Udata, Lambda, mu, alpha):
    """Performs an in-place gradient update on a column vector in `sample`.

    Parameters:
    n : int, Index of the column in `sample` to update.
    sample : np.array, The matrix containing latent vectors to be updated.
    Vsample : np.array, The matrix containing latent vectors for the related entity.
    Udata : csc_matrix, Sparse matrix with observed interactions in CSC format.
    Lambda : np.array, Precision matrix for regularization.
    mu : np.array, Mean vector for regularization.
    alpha : float, Regularization coefficient.
    """
    # Extract the nth column vector from `sample`
    un = sample[:, n]

    # Get indices and values for column n in Udata
    idx = slice(Udata.indptr[n], Udata.indptr[n + 1])  # Adjusted for 0-based indexing
    ff = Udata.indices[idx]  # Row indices
    rr = Udata.data[idx]  # Non-zero values

    # Extract columns from Vsample corresponding to the row indices in `Udata`
    MM = Vsample[:, ff]

    # Update the column vector in `sample`
    sample[:, n] += alpha * (MM @ rr - (MM @ (MM.T @ un))) + Lambda @ (mu - un)


def grad_yhat(U, V, colptr, rowval, resid, Lambda, mu, alpha):
    """Computes the gradient for yhat based on latent factor matrices U and V.

    Parameters:
    U : np.ndarray, Latent matrix for one entity.
    V : np.ndarray, Latent matrix for the related entity.
    colptr : np.ndarray, Column pointer array for sparse data structure.
    rowval : np.ndarray, Row indices corresponding to non-zero values.
    resid : np.ndarray, Residual vector for observed values.
    Lambda : np.ndarray, Regularization matrix.
    mu : np.ndarray, Mean vector for regularization.
    alpha : float, Regularization coefficient.

    Returns:
    np.ndarray : The computed gradient matrix.
    """
    num_latent = U.shape[0]
    grad = Lambda @ (mu[:, None] - U)  # Initialize grad with regularization term

    # Compute additional gradient based on V and residuals
    for n in range(U.shape[1]):
        for i in range(colptr[n], colptr[n + 1]):
            err_i = alpha * resid[i]
            v_i = rowval[i]
            for k in range(num_latent):
                grad[k, n] += err_i * V[k, v_i]
    return grad


def grad_yhat_inplace(grad, U, V, colptr, rowval, resid, lambda_vec, mu, alpha):
    """Computes the in-place gradient for yhat and updates `grad`.

    Parameters:
    grad : np.ndarray, Gradient matrix to be updated.
    U : np.ndarray, Latent matrix for one entity.
    V : np.ndarray, Latent matrix for the related entity.
    colptr : np.ndarray, Column pointer array for sparse data structure.
    rowval : np.ndarray, Row indices corresponding to non-zero values.
    resid : np.ndarray, Residual vector for observed values.
    lambda_vec : np.ndarray, Regularization vector.
    mu : np.ndarray, Mean vector for regularization.
    alpha : float, Regularization coefficient.
    """
    num_latent = U.shape[0]

    # Initialize gradient with regularization term
    for n in range(U.shape[1]):
        for k in range(num_latent):
            grad[k, n] = (U[k, n] - mu[k]) * lambda_vec[k]

    # Additional gradient update based on residuals and V matrix
    for n in range(U.shape[1]):
        for i in range(colptr[n], colptr[n + 1]):
            err_i = alpha * resid[i]
            v_i = rowval[i]
            for k in range(num_latent):
                grad[k, n] += err_i * V[k, v_i]
