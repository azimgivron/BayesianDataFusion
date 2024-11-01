import numpy as np


def copyto(F, pids):
    """Placeholder function that returns the input F unchanged.

    Parameters:
    F : Any
        The input object to be returned.
    pids : Vector
        A list of process IDs (not used in this implementation).

    Returns:
    F : Any
        The unchanged input object.
    """
    return F


def AtA_mul_B(y, F, x, lambd):
    """Computes y = (F'F + lambda * I) * x.

    Parameters:
    y : np.ndarray
        The result vector to be modified in place.
    F : np.ndarray
        The matrix F used in the computation.
    x : np.ndarray
        The vector to multiply with the matrix.
    lambd : float
        The regularization parameter.

    Raises:
    ValueError: If the lengths of y and x do not match.
    """
    if len(y) != len(x):
        raise ValueError(f"length(y)={len(y)} must equal length(x)={len(x)}")

    # Perform F' * F * x and store in y
    y[:] = np.dot(F.T, np.dot(F, x))  # Assuming At_mul_B is defined as the dot product
    y += lambd * x


def nonshared(A):
    """Returns the input A unchanged.

    Parameters:
    A : Any
        The input object to be returned.

    Returns:
    A : Any
        The unchanged input object.
    """
    return A


def normsq(x):
    """Computes the squared norm of a vector.

    Parameters:
    x : np.ndarray
        The input vector.

    Returns:
    float
        The squared norm of the vector.
    """
    return np.dot(x, x)


def norm2(x):
    """Computes the Euclidean norm (L2 norm) of a vector.

    Parameters:
    x : np.ndarray
        The input vector.

    Returns:
    float
        The Euclidean norm of the vector.
    """
    return np.sqrt(normsq(x))


def prod_add(p, mult, r):
    """Computes p += mult * r, where the operation is done element-wise.

    Parameters:
    p : np.ndarray
        The first vector to be updated.
    mult : float
        The multiplier for vector r.
    r : np.ndarray
        The vector to be multiplied and added to p.
    """
    p += mult * r  # Element-wise operation using NumPy broadcasting


def add_prod(x, mult, v):
    """Computes x += mult * v, where the operation is done element-wise.

    Parameters:
    x : np.ndarray
        The vector to be updated.
    mult : float
        The multiplier for vector v.
    v : np.ndarray
        The vector to be multiplied and added to x.
    """
    x += mult * v  # Element-wise operation using NumPy broadcasting


def sub_prod(x, mult, v):
    """Computes x -= mult * v, where the operation is done element-wise.

    Parameters:
    x : np.ndarray
        The vector to be updated.
    mult : float
        The multiplier for vector v.
    v : np.ndarray
        The vector to be multiplied and subtracted from x.
    """
    x -= mult * v  # Element-wise operation using NumPy broadcasting


def cg_AtA_ref(Aref, b, lambd, tol, maxiter=None):
    """Performs conjugate gradient method for solving Ax = b with a regularization term,
    using a future for A.

    Parameters:
    Aref : Future
        A future that resolves to the matrix A.
    b : np.ndarray
        The right-hand side vector.
    lambd : float
        The regularization parameter.
    tol : float
        The tolerance for stopping criteria.
    maxiter : int, optional
        The maximum number of iterations.

    Returns:
    np.ndarray
        The solution vector.
    """
    A = Aref.result()  # Assuming Aref is a future that resolves to the matrix A
    return cg_AtA(A, b, lambd, tol=tol, maxiter=maxiter)


def cg_AtA(A, b, lambd, tol=None, maxiter=None):
    """Performs conjugate gradient method for solving Ax = b with a regularization term.

    Parameters:
    A : np.ndarray
        The matrix A in the equation.
    b : np.ndarray
        The right-hand side vector.
    lambd : float
        The regularization parameter.
    tol : float, optional
        The tolerance for stopping criteria.
    maxiter : int, optional
        The maximum number of iterations.

    Returns:
    np.ndarray
        The solution vector.
    """
    if tol is None:
        tol = A.shape[1] * np.finfo(float).eps
    if maxiter is None:
        maxiter = A.shape[1]

    return cg_AtA_internal(
        A, b, lambd, np.zeros(len(b)), np.zeros(len(b)), tol=tol, maxiter=maxiter
    )


def cg_AtA_internal(A, b, lambd, p, z, tol=None, maxiter=None):
    """Internal function for the conjugate gradient method.

    Parameters:
    A : np.ndarray
        The matrix A in the equation.
    b : np.ndarray
        The right-hand side vector.
    lambd : float
        The regularization parameter.
    p : np.ndarray
        The current search direction vector.
    z : np.ndarray
        The auxiliary vector for the algorithm.
    tol : float, optional
        The tolerance for stopping criteria.
    maxiter : int, optional
        The maximum number of iterations.

    Returns:
    np.ndarray
        The solution vector.
    """
    if tol is None:
        tol = A.shape[1] * np.finfo(float).eps
    if maxiter is None:
        maxiter = A.shape[1]

    tol *= np.linalg.norm(b)

    x = np.zeros(len(b))
    r = np.copy(b)  # r = b - A * x, but x is initially zero, so r = b
    np.copyto(p, r)
    bkden = 0.0

    for iter in range(maxiter):
        bknum = np.dot(r, r)  # normsq(r)
        err = np.sqrt(bknum)

        if err < tol:
            return x  # or return x, err, iter

        if iter > 0:
            bk = bknum / bkden
            prod_add(p, bk, r)  # p += bk * r

        bkden = bknum

        # z = (A.T @ A + lambda * I) * p
        z = AtA_mul_B(A, p, lambd)

        ak = bknum / np.dot(z, p)

        x += ak * p  # x += ak * p
        r -= ak * z  # r -= ak * z

    return x  # or return x, np.sqrt(normsq(r)), maxiter


def AtA_mul_B(A, p, lambd):
    """Computes y = (A.T @ A + lambda * I) @ p.

    Parameters:
    A : np.ndarray
        The matrix A in the equation.
    p : np.ndarray
        The vector to multiply with the matrix.
    lambd : float
        The regularization parameter.

    Returns:
    np.ndarray
        The result of the multiplication.
    """
    return np.dot(A.T, np.dot(A, p)) + lambd * p


def prod_add(p, mult, r):
    """Computes p += mult * r.

    Parameters:
    p : np.ndarray
        The first vector to be updated.
    mult : float
        The multiplier for vector r.
    r : np.ndarray
        The vector to be multiplied and added to p.
    """
    p += mult * r


def add_prod(x, mult, v):
    """Computes x += mult * v.

    Parameters:
    x : np.ndarray
        The vector to be updated.
    mult : float
        The multiplier for vector v.
    v : np.ndarray
        The vector to be multiplied and added to x.
    """
    x += mult * v


def sub_prod(x, mult, v):
    """Computes x -= mult * v.

    Parameters:
    x : np.ndarray
        The vector to be updated.
    mult : float
        The multiplier for vector v.
    v : np.ndarray
        The vector to be multiplied and subtracted from x.
    """
    x -= mult * v
