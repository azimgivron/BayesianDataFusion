from time import time

import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix


class VBModel:
    def __init__(
        self,
        num_latent,
        N,
        mu_u=None,
        Euu=None,
        nu_N=None,
        W_N=None,
        mu_N=None,
        b_N=None,
        Winv_0=None,
        mu_0=None,
        b_0=None,
    ):
        """Initializes a Variational Bayes Model (VBModel) with default values.

        Parameters:
        num_latent : int, Number of latent dimensions.
        N : int, Number of instances.
        mu_u : np.array, Optional means for u_i (default: random).
        Euu : np.array, Optional 3D array for E[u u'] values (default: computed from mu_u).
        nu_N : float, Optional Nu parameter for Normal-Wishart (default: num_latent + N).
        W_N : np.array, Optional Matrix W in Normal-Wishart (default: I/N).
        mu_N : np.array, Optional mean mu for Normal-Wishart (default: zeros).
        b_N : float, Optional Beta parameter for Normal-Wishart (default: 2.0 + N).
        Winv_0 : np.array, Optional inverted hyperprior W_0 (default: identity).
        mu_0 : np.array, Optional hyperprior mean mu_0 (default: zeros).
        b_0 : float, Optional hyperprior beta (default: 2.0).
        """
        # Set default values if not provided
        self.mu_u = mu_u if mu_u is not None else np.random.randn(num_latent, N)
        self.Euu = Euu if Euu is not None else np.zeros((num_latent, num_latent, N))
        self.nu_N = nu_N if nu_N is not None else num_latent + N
        self.W_N = W_N if W_N is not None else np.eye(num_latent) / N
        self.mu_N = mu_N if mu_N is not None else np.zeros(num_latent)
        self.b_N = b_N if b_N is not None else 2.0 + N
        self.Winv_0 = Winv_0 if Winv_0 is not None else np.eye(num_latent)
        self.mu_0 = mu_0 if mu_0 is not None else np.zeros(num_latent)
        self.b_0 = b_0 if b_0 is not None else 2.0

        # Populate Euu with inv(W_N) + mu_u[:, n] * mu_u[:, n].T for each instance
        for n in range(N):
            self.Euu[:, :, n] = np.linalg.inv(self.W_N) + np.outer(
                self.mu_u[:, n], self.mu_u[:, n]
            )


def bpmf_vb(data, num_latent=10, verbose=True, niter=100, clamp=[]):
    """Bayesian Probabilistic Matrix Factorization with Variational Bayes.

    Parameters:
    data : RelationData, The input data with relation and entity information.
    num_latent : int, Number of latent dimensions.
    verbose : bool, Flag to enable verbose output.
    niter : int, Number of iterations for the algorithm.
    clamp : list of float, Optional clamp values.

    Returns:
    dict : A dictionary containing Umodel, Vmodel, rmse, rmse_train, and alpha.
    """
    # Initialization
    rmse = np.nan
    rmse_train = np.nan
    Umodel = VBModel(num_latent, data.entities[0].count)
    Vmodel = VBModel(num_latent, data.entities[1].count)

    # Data setup
    df = data.relations[0].data.df
    mean_value = np.mean(df.iloc[:, -1])
    uid = np.array(df.iloc[:, 0], dtype=np.int32)
    vid = np.array(df.iloc[:, 1], dtype=np.int32)
    val = np.array(df.iloc[:, 2]) - mean_value
    Udata = coo_matrix(
        (val, (vid, uid)), shape=(data.entities[1].count, data.entities[0].count)
    ).tocsc()
    Vdata = Udata.transpose()

    test_uid = np.array(data.relations[0].test_vec.iloc[:, 0])
    test_vid = np.array(data.relations[0].test_vec.iloc[:, 1])
    test_val = np.array(data.relations[0].test_vec.iloc[:, 2])

    alpha = data.relations[0].model.alpha

    for i in range(1, niter + 1):
        time0 = time()
        update_u(Umodel, Vmodel, Udata, alpha)
        update_u(Vmodel, Umodel, Vdata, alpha)

        update_prior(Umodel)
        update_prior(Vmodel)

        yhat = clamp_values(predict(Umodel, Vmodel, mean_value, test_uid, test_vid), clamp)
        rmse = np.sqrt(np.mean((yhat - test_val) ** 2))
        yhat_train = clamp_values(predict(Umodel, Vmodel, mean_value, uid, vid), clamp)
        rmse_train = np.sqrt(np.mean((yhat_train - mean_value - val) ** 2))

        time1 = time()

        if verbose:
            print(
                f"{i:3d}: |U|={np.linalg.norm(Umodel.mu_u):.4e}  |V|={np.linalg.norm(Vmodel.mu_u):.4e}  "
                f"RMSE={rmse:.4f}  RMSE(train)={rmse_train:.4f}  [took {time1 - time0:.2f}s]"
            )

    return {
        "Umodel": Umodel,
        "Vmodel": Vmodel,
        "rmse": rmse,
        "rmse_train": rmse_train,
        "alpha": alpha,
    }


def clamp_values(predictions, clamp):
    """Clamps predictions within a specified range.

    Parameters:
    predictions : np.array, The predicted values to clamp.
    clamp : list, A two-element list with min and max values for clamping.

    Returns:
    np.array : The clamped predictions.
    """
    if len(clamp) == 2:
        return np.clip(predictions, clamp[0], clamp[1])
    return predictions


def add(X, Y, dim, mult):
    """Adds `mult * Y[:,:,dim]` to matrix X in-place.

    Parameters:
    X : np.ndarray, 2D matrix to be updated.
    Y : np.ndarray, 3D array with the same first two dimensions as X.
    dim : int, The third dimension index in Y to use for the addition.
    mult : float, The multiplier applied to Y[:,:,dim] before adding.

    Raises:
    ValueError : If X and Y do not have the same dimensions in the first two axes.
    """
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same size in the first and second dimensions.")

    X += mult * Y[:, :, dim]


def update_u(Umodel, Vmodel, Udata, alpha):
    """Updates the Umodel in-place using variational Bayes update rules.

    Parameters:
    Umodel : VBModel, The variational model for U.
    Vmodel : VBModel, The variational model for V.
    Udata : csc_matrix, Sparse data matrix for U.
    alpha : float, Scaling factor for updates.
    """
    A = Umodel.W_N * Umodel.nu_N
    b = Umodel.W_N @ (Umodel.nu_N * Umodel.mu_N)
    num_latent = Umodel.mu_u.shape[0]
    colptr = Udata.indptr
    rowval = Udata.indices
    nzval = Udata.data

    for uu in range(Umodel.mu_u.shape[1]):
        # Indices for non-zero entries in the column
        idx = slice(colptr[uu], colptr[uu + 1])
        ff = rowval[idx]
        rr = nzval[idx]

        # Initialize L with A
        L = A.copy()
        for vv in ff:
            add(
                L, Vmodel.Euu, vv, alpha
            )  # This function applies L += alpha * Vmodel.Euu[:, :, vv]

        # Calculate inverse of L
        Linv = inv(L)

        # Compute mu and update Umodel
        MM = Vmodel.mu_u[:, ff]
        mu = Linv @ (b + alpha * MM @ rr)

        Umodel.mu_u[:, uu] = mu
        Umodel.Euu[:, :, uu] = Linv + np.outer(mu, mu)


def update_prior(m):
    """Updates the prior parameters in the VBModel in-place.

    Parameters:
    m : VBModel, The variational Bayes model whose prior parameters are updated.
    """
    num_latent = m.mu_u.shape[0]

    # Update mu_N
    m.mu_N = (m.b_0 * m.mu_0 + np.sum(m.mu_u, axis=1)) / m.b_N

    # Update W_N
    sum_Euu = np.sum(m.Euu, axis=2).reshape(num_latent, num_latent)
    m.W_N = inv(
        m.Winv_0 + sum_Euu + m.b_0 * np.outer(m.mu_0, m.mu_0) - m.b_N * np.outer(m.mu_N, m.mu_N)
    )


def predict(Umodel, Vmodel, mean_value, uids, vids):
    """Predicts values based on the provided Umodel and Vmodel.

    Parameters:
    Umodel : VBModel, The variational model for U.
    Vmodel : VBModel, The variational model for V.
    mean_value : float, The baseline mean value for predictions.
    uids : list or np.array, Indices of U model.
    vids : list or np.array, Indices of V model.

    Returns:
    np.array: The predicted values for each pair of uids and vids.
    """
    yhat = np.full(len(uids), mean_value)  # Initialize with mean_value
    for i in range(len(uids)):
        yhat[i] += np.dot(Umodel.mu_u[:, uids[i]], Vmodel.mu_u[:, vids[i]])
    return yhat


def show_VBModel(m):
    """Displays summary information about the VBModel.

    Parameters:
    m : VBModel, The variational Bayes model.
    """
    print(f"VBModel of {m.mu_u.shape[1]} instances: |mu_u|={np.linalg.norm(m.mu_u):.3e}")
