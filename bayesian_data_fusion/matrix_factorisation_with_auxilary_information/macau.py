import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def macau(
    data,
    num_latent=10,
    lambda_beta=np.nan,
    burnin=500,
    psamples=200,
    verbose=True,
    full_lambda_u=True,
    reset_model=True,
    compute_ff_size=6500,
    latent_pids=None,
    latent_blas_threads=1,
    cg_pids=None,
    full_prediction=False,
    rmse_train=False,
    tol=np.nan,
    output="",
    output_beta=False,
    output_type="csv",
    clamp=[],
    f=None,
):
    """Main function to run the Macau algorithm with specified parameters.

    Parameters:
    data : RelationData, The relation data object.
    num_latent : int, Number of latent features.
    lambda_beta : float, Regularization parameter for beta.
    burnin : int, Number of burn-in samples.
    psamples : int, Number of posterior samples to collect.
    verbose : bool, Enable verbose output.
    full_lambda_u : bool, Enable full lambda_u estimation.
    reset_model : bool, Whether to reset the model before sampling.
    compute_ff_size : int, Size for computing FF.
    latent_pids : list, Process IDs for parallel latent vector sampling.
    latent_blas_threads : int, Number of BLAS threads.
    cg_pids : list, Process IDs for CG tasks.
    full_prediction : bool, Enable full prediction.
    rmse_train : bool, Calculate RMSE on training set.
    tol : float, Tolerance for beta update.
    output : str, Output prefix for saving samples.
    output_beta : bool, Save beta samples to disk.
    output_type : str, Format for output ("csv" or "binary").
    clamp : list, Clamp range for predictions.
    f : function, Custom function for additional output.

    Returns:
    dict : Contains model metrics, predictions, and optionally saved samples.
    """
    if latent_pids is None:
        latent_pids = []
    if cg_pids is None:
        cg_pids = []

    if reset_model:
        reset_model_fn(data, num_latent, lambda_beta, compute_ff_size, cg_pids)

    if verbose:
        print("Model setup complete.")

    result = gibbs_sampling_loop(
        data,
        num_latent,
        lambda_beta,
        burnin,
        psamples,
        verbose,
        full_lambda_u,
        full_prediction,
        rmse_train,
        tol,
        output,
        output_beta,
        output_type,
        clamp,
        f,
        latent_pids,
        latent_blas_threads,
    )
    return result


def reset_model_fn(data, num_latent, lambda_beta, compute_ff_size, cg_pids):
    """Resets the model if specified in the parameters."""
    reset(
        data, num_latent, lambda_beta=lambda_beta, compute_ff_size=compute_ff_size, cg_pids=cg_pids
    )


def gibbs_sampling_loop(
    data,
    num_latent,
    lambda_beta,
    burnin,
    psamples,
    verbose,
    full_lambda_u,
    full_prediction,
    rmse_train,
    tol,
    output,
    output_beta,
    output_type,
    clamp,
    f,
    latent_pids,
    latent_blas_threads,
):
    """Main loop for Gibbs sampling with burn-in and posterior samples.

    Parameters:
    data : RelationData, The relation data object.
    num_latent : int, Number of latent features.
    lambda_beta : float, Regularization parameter for beta.
    burnin : int, Number of burn-in samples.
    psamples : int, Number of posterior samples to collect.
    verbose : bool, Enable verbose output.
    full_lambda_u : bool, Enable full lambda_u estimation.
    full_prediction : bool, Enable full prediction.
    rmse_train : bool, Calculate RMSE on training set.
    tol : float, Tolerance for beta update.
    output : str, Output prefix for saving samples.
    output_beta : bool, Save beta samples to disk.
    output_type : str, Format for output ("csv" or "binary").
    clamp : list, Clamp range for predictions.
    f : function, Custom function for additional output.
    latent_pids : list, Process IDs for parallel latent vector sampling.
    latent_blas_threads : int, Number of BLAS threads.

    Returns:
    dict : Final results including performance metrics and predictions.
    """
    latent_multi_threading, latent_data_refs = setup_multi_threading(
        data, latent_pids, latent_blas_threads, verbose
    )

    probe_rat_all, counter_prob, f_output = None, 1, []
    err_avg, rmse_avg = 0.0, 0.0
    yhat_full = np.zeros(data.relations[0].size()) if full_prediction else None

    for i in range(1, burnin + psamples + 1):
        start_time = time.time()

        # Sample model parameters (alpha, beta, latent vectors)
        sample_relation_model(data)
        sample_latent_vectors(
            data, latent_multi_threading, latent_data_refs, num_latent, full_lambda_u
        )

        # Collect predictions after burn-in
        if i > burnin:
            collect_samples(
                data,
                output,
                output_type,
                output_beta,
                psamples,
                i,
                burnin,
                yhat_full,
                rmse_train,
                clamp,
                f,
                f_output,
            )

        # Output diagnostics
        end_time = time.time()
        if verbose:
            print_iteration_status(i, err_avg, rmse_avg, end_time - start_time)

    return compile_results(
        data, num_latent, burnin, psamples, rmse_train, full_prediction, clamp, f_output, yhat_full
    )


def setup_multi_threading(data, latent_pids, latent_blas_threads, verbose):
    """Configures multi-threading if applicable."""
    latent_multi_threading, latent_data_refs = False, None
    if len(latent_pids) >= 1 and len(data.relations) == 1 and not data.relations[0].has_features():
        latent_multi_threading = True
        if verbose:
            print(f"Setting up multi-threaded sampling with {len(latent_pids)} threads.")
        fastidf = FastIDF(data.relations[0].data)
        with ProcessPoolExecutor() as executor:
            latent_data_refs = [executor.submit(fastidf.fetch) for _ in latent_pids]
    return latent_multi_threading, latent_data_refs


def sample_relation_model(data):
    """Samples alpha and beta parameters for the relations."""
    for r in data.relations:
        if r.model.alpha_sample:
            error = r.pred() - r.data.get_values()
            r.model.alpha = sample_alpha(r.model.alpha_lambda0, r.model.alpha_nu0, error)
        if r.has_features():
            r.model.beta = sample_beta_rel(r)
            r.temp.linear_values = r.model.mean_value + r.F @ r.model.beta


def sample_latent_vectors(
    data, latent_multi_threading, latent_data_refs, num_latent, full_lambda_u
):
    """Samples the latent vectors for the entities."""
    for en in data.entities:
        mj = en.model
        if latent_multi_threading:
            sample_latent_all(data.relations[0], latent_data_refs, en, mj)
        else:
            single_thread_sample_latent(en, mj, full_lambda_u)


def single_thread_sample_latent(en, mj, full_lambda_u):
    """Samples latent vectors in a single-threaded mode."""
    if en.has_features():
        mj.uhat = F_mul_beta(en)
        mu_matrix = mj.mu + mj.uhat
        sample_user2_all(en, mu_matrix, en.modes, en.modes_other)
    else:
        sample_user2_all(en, en.modes, en.modes_other)
    update_prior_latent(mj, en, full_lambda_u)


def update_prior_latent(mj, en, full_lambda_u):
    """Updates the prior for the latent variables."""
    U = mj.sample - (mj.uhat if en.has_features() else 0)
    nu = mj.nu0 + (mj.beta.shape[0] if full_lambda_u else 0)
    Tinv = mj.WI + (mj.beta.T @ mj.beta * en.lambda_beta if full_lambda_u else 0)
    mj.mu, mj.Lambda = rand(ConditionalNormalWishart(U, mj.mu0, mj.b0, Tinv, nu))


def collect_samples(
    data,
    output,
    output_type,
    output_beta,
    psamples,
    i,
    burnin,
    yhat_full,
    rmse_train,
    clamp,
    f,
    f_output,
):
    """Collects posterior samples and predictions."""
    rel = data.relations[0]
    probe_rat = pred(rel, rel.test_vec, rel.test_F)

    if yhat_full is not None:
        yhat_full += pred_all(data.relations[0])

    if output:
        save_samples(data.entities, output, i, burnin, output_type, output_beta)


def print_iteration_status(i, err_avg, rmse_avg, elapsed_time):
    """Prints the status for each iteration."""
    print(
        f"Iteration {i}, Error avg: {err_avg:.4f}, RMSE avg: {rmse_avg:.4f}, Time: {elapsed_time:.2f} sec"
    )


def compile_results(
    data, num_latent, burnin, psamples, rmse_train, full_prediction, clamp, f_output, yhat_full
):
    """Compiles final results after sampling is complete."""
    result = {
        "num_latent": num_latent,
        "burnin": burnin,
        "psamples": psamples,
        "lambda_beta": data.entities[0].lambda_beta,
        "RMSE": 0.0,  # Placeholder
        "accuracy": 0.0,  # Placeholder
        "ROC": 0.0,  # Placeholder for actual ROC calculation
    }
    if full_prediction:
        result["predictions_full"] = yhat_full / psamples
    return result
