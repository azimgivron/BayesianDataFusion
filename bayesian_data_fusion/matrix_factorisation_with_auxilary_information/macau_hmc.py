import numpy as np


class HMCModel:
    def __init__(self, num_latent, N, Ldiag):
        """Initializes the HMCModel with momentum and a diagonal mass matrix.

        Parameters:
        num_latent : int, Number of latent dimensions.
        N : int, Number of instances.
        Ldiag : list of float, Diagonal elements for the mass matrix.
        """
        self.momentum = np.zeros((num_latent, N))  # Initialize momentum
        self.G = np.tile(Ldiag, (1, N))  # Create mass matrix (G) as a 2D array

    def deepcopy(self):
        """Creates a deep copy of the HMCModel instance."""
        return HMCModel.from_existing(self.momentum.copy())

    @classmethod
    def from_existing(cls, momentum):
        """Alternative constructor for creating an HMCModel from existing momentum."""
        model = cls.__new__(cls)  # Create a new instance without calling __init__
        model.momentum = momentum
        return model


def macau_hmc(
    data,
    num_latent=10,
    verbose=True,
    burnin=100,
    psamples=100,
    L=10,
    L_inner=1,
    prior_freq=8,
    eps=0.01,
    reset_model=True,
    clamp=[],
):
    """Hamiltonian Monte Carlo (HMC) method for Bayesian Probabilistic Matrix
    Factorization (BPMF).

    Parameters:
    data : RelationData, Input data for the model.
    num_latent : int, Number of latent dimensions.
    verbose : bool, If True, prints progress messages.
    burnin : int, Number of burn-in iterations.
    psamples : int, Number of samples to draw.
    L : int, Number of leapfrog steps.
    L_inner : int, Number of inner leapfrog steps.
    prior_freq : int, Frequency of prior updates.
    eps : float, Step size for HMC.
    reset_model : bool, If True, resets the model.
    clamp : list of float, Values to clamp predictions.

    Returns:
    dict : Contains RMSE and other metrics.
    """
    # Initialization
    if verbose:
        print("Model setup")
    if reset_model:
        reset(data, num_latent)

    rmse = np.nan
    rmse_sample = np.nan
    rmse_train = np.nan
    Umodel = HMCModel(num_latent, data.entities[0].count, np.diag(data.entities[0].model.Lambda))
    Vmodel = HMCModel(num_latent, data.entities[1].count, np.diag(data.entities[1].model.Lambda))

    # Data preparation
    df = data.relations[0].data.df
    mean_value = np.mean(df[:, -1])
    uid = df[:, 0].astype(np.int32)
    vid = df[:, 1].astype(np.int32)
    val = df[:, 2].astype(np.float64) - mean_value
    Udata = sparse(vid, uid, val, data.entities[1].count, data.entities[0].count)
    Vdata = Udata.T

    test_uid = data.relations[0].test_vec[:, 0].astype(int)
    test_vid = data.relations[0].test_vec[:, 1].astype(int)
    test_val = data.relations[0].test_vec[:, 2]
    test_idx = np.column_stack((test_uid, test_vid))

    alpha = data.relations[0].model.alpha
    yhat_post = np.zeros(len(test_val))

    for i in range(1, burnin + psamples + 1):
        time0 = time.time()

        if i == burnin + 1:
            if verbose:
                print("================== Burnin complete ===================")

        if verbose:
            print(f"======= Step {i} =======")
            print(f"eps = {eps:.2e}")

        # HMC sampling momentum
        sample(Umodel)
        sample(Vmodel)
        kinetic_start = computeKinetic(Umodel) + computeKinetic(Vmodel)
        potential_start = computePotential(uid, vid, val, data.relations[0])
        Ustart = np.copy(data.entities[0].model.sample)
        Vstart = np.copy(data.entities[1].model.sample)

        # Follow Hamiltonians
        hmc_update_u(
            Umodel, Vmodel, Udata, alpha, data.entities[0], data.entities[1], L_inner, eps / 2
        )
        for l in range(1, L + 1):
            hmc_update_u(
                Vmodel, Umodel, Vdata, alpha, data.entities[1], data.entities[0], L_inner, eps
            )
            if l < L:
                hmc_update_u(
                    Umodel, Vmodel, Udata, alpha, data.entities[0], data.entities[1], L_inner, eps
                )

            if verbose:
                print(
                    f"  Momentum {l}: |r_U| = {np.linalg.norm(Umodel.momentum):.4e}, |r_V| = {np.linalg.norm(Vmodel.momentum):.4e}"
                )

        hmc_update_u(
            Umodel, Vmodel, Udata, alpha, data.entities[0], data.entities[1], L_inner, eps / 2
        )
        if verbose:
            print(
                f"  Momentum L: |r_U| = {np.linalg.norm(Umodel.momentum):.4e}, |r_V| = {np.linalg.norm(Vmodel.momentum):.4e}"
            )

        # Verify p(U,V) is fine
        kinetic_final = computeKinetic(Umodel) + computeKinetic(Vmodel)
        potential_final = computePotential(uid, vid, val, data.relations[0])

        dH = potential_start - potential_final + kinetic_start - kinetic_final
        if verbose:
            print(
                f"  ΔH = {dH:.4e}  ΔKin = {kinetic_final - kinetic_start:.4e}  ΔPot = {potential_final - potential_start:.4e}"
            )

        if np.random.rand() < np.exp(dH):
            # Accept
            if verbose:
                print("-> ACCEPTED!")
        else:
            # Reject
            if verbose:
                print("-> REJECTED!")
            data.entities[0].model.sample = Ustart
            data.entities[1].model.sample = Vstart
            if dH < -6:
                # Decrease eps by 2
                neweps = eps / 2
                newL = int(np.ceil(L * 1.6))
                if verbose:
                    print(f"Reducing eps from {eps:.2e} to {neweps:.2e}.")
                    print(f"Increasing L from {L} to {newL}.")
                eps = neweps
                L = newL

        # Sampling prior for latents
        if i % prior_freq == 0:
            if verbose:
                print("Updating priors...")
            for en in data.entities:
                update_latent_prior(en, True)

        rel = data.relations[0]
        yhat_raw = pred(rel, test_idx, rel.test_F)
        yhat = clamp(yhat_raw, clamp)
        update_yhat_post(yhat_post, yhat_raw, i, burnin)

        rmse = np.sqrt(np.mean((yhat - test_val) ** 2))
        rmse_post = np.sqrt(np.mean((makeClamped(yhat_post, clamp) - test_val) ** 2))

        time1 = time.time()

        if verbose:
            print(
                f"{i:3d}: |U|={np.linalg.norm(data.entities[0].model.sample):.4e}  |V|={np.linalg.norm(data.entities[1].model.sample):.4e}  RMSE={rmse:.4f}  RMSE(avg)={rmse_post:.4f} [took {time1 - time0:.2f}s]"
            )

    return {"rmse": rmse, "rmse_train": rmse_train, "alpha": alpha}


def hmc_update_u(U_hmcmodel, V_hmcmodel, Udata, alpha, en, enV, L, eps):
    """Performs Hamiltonian Monte Carlo update for the latent variable U.

    Parameters:
    U_hmcmodel : HMCModel, The HMC model for U.
    V_hmcmodel : HMCModel, The HMC model for V.
    Udata      : SparseMatrixCSC, The sparse matrix data for U.
    alpha      : float, The alpha parameter for the model.
    en         : Entity, The entity representing U.
    enV        : Entity, The entity representing V.
    L          : int, Number of full steps.
    eps        : float, Step size for the update.
    """
    model = en.model
    sample = model.sample
    Vsample = enV.model.sample
    momentum = U_hmcmodel.momentum
    num_latent = momentum.shape[0]

    # 1) Make half step in momentum
    subtract_grad(momentum, model, sample, Vsample, Udata, alpha, eps / 2)

    # 2) L times: make full steps with U, full step with momentum (except for last step)
    for i in range(1, L + 1):
        add(sample, momentum, eps)
        if i < L:
            subtract_grad(momentum, model, sample, Vsample, Udata, alpha, eps)

    # 3) Make half step in momentum
    subtract_grad(momentum, model, sample, Vsample, Udata, alpha, eps / 2)


def add(X, Y, mult):
    """Computes X += mult * Y.

    Parameters:
    X : np.ndarray, The first matrix (modified in place).
    Y : np.ndarray, The second matrix.
    mult : float, The multiplier.

    Raises:
    ValueError: If X and Y do not have the same size.
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same size.")

    # Using in-place addition with broadcasting
    X += mult * Y


def subtract_grad(momentum, model, sample, Vsample, Udata, alpha, eps):
    """Updates the momentum by subtracting the gradient scaled by eps.

    Parameters:
    momentum : np.ndarray, The momentum matrix to be updated.
    model : EntityModel, The model containing Lambda and mu.
    sample : np.ndarray, The sample matrix.
    Vsample : np.ndarray, The V sample matrix.
    Udata : SparseMatrixCSC, The sparse matrix containing data.
    alpha : float, The scaling factor.
    eps : float, The step size for the gradient update.
    """
    num_latent = momentum.shape[0]

    for n in range(sample.shape[1]):  # Loop over columns of sample
        tmp = grad(
            n, sample, Vsample, Udata, model.Lambda, model.mu, alpha
        )  # Calculate the gradient
        for k in range(num_latent):  # Loop over latent dimensions
            momentum[k, n] -= eps * tmp[k]  # Update the momentum


def grad(n, sample, Vsample, Udata, Lambda, mu, alpha):
    """Computes the gradient for the n-th sample.

    Parameters:
    n : int, The index of the sample.
    sample : np.ndarray, The sample matrix.
    Vsample : np.ndarray, The V sample matrix.
    Udata : SparseMatrixCSC, The sparse data matrix.
    Lambda : np.ndarray, The Lambda matrix.
    mu : np.ndarray, The mu vector.
    alpha : float, The scaling factor.

    Returns:
    np.ndarray: The computed gradient.
    """
    un = sample[:, n]  # Get the n-th column of the sample
    idx = slice(Udata.colptr[n], Udata.colptr[n + 1])  # Create a slice for indices
    ff = Udata.rowval[idx]  # Row values from Udata
    rr = Udata.nzval[idx]  # Non-zero values from Udata

    MM = Vsample[:, ff]  # Get the corresponding columns from Vsample

    # Compute the gradient
    gradient = -alpha * (MM @ rr - (MM @ MM.T) @ un) - Lambda @ (mu - un)

    return gradient


def computeKinetic(m):
    """Computes the kinetic energy based on the HMC model.

    Parameters:
    m : HMCModel, The HMC model containing momentum and G.

    Returns:
    float: The computed kinetic energy.
    """
    kin = 0.0
    momentum = m.momentum
    G = m.G

    # Compute the kinetic energy term: r'*G*r + log|G|
    for i in range(len(momentum)):
        kin += momentum[i] * momentum[i] * G[i] + np.log(G[i])

    return 0.5 * kin


def compute_potential(uid, vid, val, relation):
    """Computes the negative log likelihood of the data and the prior for the models.

    Parameters:
    uid : list or np.ndarray, User IDs.
    vid : list or np.ndarray, Item IDs.
    val : list or np.ndarray, Observed values.
    relation : Relation, The relation object containing the models.

    Returns:
    float: The computed potential energy.
    """
    alpha = relation.model.alpha
    Umodel = relation.entities[0].model  # Adjusted for 0-based index
    Vmodel = relation.entities[1].model
    Usample = Umodel.sample
    Vsample = Vmodel.sample
    energy = 0.0

    # Data energy calculation
    for i in range(len(uid)):
        energy += (column_dot(Usample, Vsample, uid[i], vid[i]) - val[i]) ** 2

    energy *= alpha / 2

    # Priors
    # sum(u_i' * Lambda_u * u_i) = trace(Lambda_u * U * U')
    energy += np.sum(Umodel.Lambda * (Usample @ Usample.T)) / 2
    energy += np.sum(Vmodel.Lambda * (Vsample @ Vsample.T)) / 2

    # -2 * sum(mu_u' * Lambda_u * u_i)
    energy -= np.dot(Umodel.mu.T, Umodel.Lambda @ np.sum(Usample, axis=1))
    energy -= np.dot(Vmodel.mu.T, Vmodel.Lambda @ np.sum(Vsample, axis=1))

    return energy


def column_dot(X, Y, i, j):
    """Computes the dot product of the ith column of X and the jth column of Y.

    Parameters:
    X : np.ndarray, The first matrix.
    Y : np.ndarray, The second matrix.
    i : int, The column index in X (1-based).
    j : int, The column index in Y (1-based).

    Returns:
    float: The dot product of the specified columns.
    """
    # Adjusting for 0-based indexing in Python
    i -= 1
    j -= 1

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    if X.shape[1] <= i:
        raise ValueError("X must have at least i columns.")
    if Y.shape[1] <= j:
        raise ValueError("Y must have at least j columns.")

    d = 0.0
    for k in range(X.shape[0]):
        d += X[k, i] * Y[k, j]

    return d


def update_yhat_post(yhat_post, yhat_raw, i, burnin):
    """Updates the posterior predictions.

    Parameters:
    yhat_post : np.ndarray, The current posterior prediction.
    yhat_raw : np.ndarray, The new prediction to incorporate.
    i : int, The current iteration or sample index.
    burnin : int, The number of burn-in samples.

    Returns:
    np.ndarray: The updated posterior prediction.
    """
    # Adjusting for 0-based indexing in Python
    if i <= burnin + 1:
        np.copyto(yhat_post, yhat_raw)
        return yhat_post

    # Averaging
    n = i - burnin - 1
    yhat_post[:] = (n * yhat_post + yhat_raw) / (n + 1)

    return yhat_post
