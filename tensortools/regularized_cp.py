import tensorly
from tensorly.kruskal import kruskal_to_tensor
from tensorly.base import unfold
from tensorly.tenalg import khatri_rao
from .kruskal import standardize_factors
import numpy as np
from tqdm import trange
from scipy.optimize import check_grad

## Penalty functions ##

def no_penalty(x, scale):
    """No penalty operator
    """
    return 0.0

def l1(x, scale):
    """Calculates l1 norm of a matrix `x`
    """
    return scale * np.sum(np.abs(x))

def nucnorm(x, scale):
    """Calculates the nuclear norm of a matrix `x`
    """
    s = np.linalg.svd(x, compute_uv=False)
    return np.sum(np.abs(s))

## Proximal operators ##

def prox_no_penalty(x, lr, scale):
    """No penalty prox operator
    """
    return x

def prox_l1(x, lr, scale):
    """Proximal operator for L1 regularization
    """
    lmbda = penalty * lr
    return (x - lmbda) * (x >= lmbda) + (x + lmbda) * (x <= -lmbda)

def prox_nonneg_l1(x, lr, scale):
    """Proximal operator for L1 regularization and nonnegativity constraint
    """
    return np.maximum(0, x - lr * scale)


def prox_nucnorm(x, lr, scale):
    """Proximal operater on the nuclear norm of a matrix.
    """
    u, s, v = np.linalg.svd(x, full_matrices=False)
    sthr = np.maximum(s - (scale * lr), 0)
    return np.dot(u*sthr, v)

def prox_nonneg_nucnorm(x, lr, scale):
    """Proximal operator for nuclear norm regularization with nonnegativity constraint
    """
    return np.maximum(0, prox_nucnorm(x, lr, scale))

def sparse_cpfit(tensor, rank, penalty_scales, penalize_l1=None,
                 penalize_nucnorm=None,  niter=1000, lr=1.0,
                 nonneg=False, factors=None):
    """CP decomposition with L1 penalty on the factor matrices
    """

    # check inputs
    if penalize_l1 is None:
        penalize_l1 = [False for _ in range(tensor.ndim)]
    else:
        assert np.iterable(penalize_l1)
        assert len(penalize_l1) == tensor.ndim

    if not np.iterable(penalty_scales) or len(penalty_scales) != tensor.ndim:
        raise ValueError('Penalties should be specified as a list.')
    penalty_scales = np.array(penalty_scales)

    # initialize factors
    if factors is None and nonneg:
        factors = [np.random.rand(s, rank) for s in tensor.shape]
    elif factors is None:
        factors = [np.random.randn(s, rank) for s in tensor.shape]

    # initialize penalties
    for mode in range(tensor.ndim):
        # initialize penalty functions

    penalty_funcs = [l1 if s > 0 else no_penalty for s in penalty_scales]
    penalties = np.array([f(fctr, s) for fctr, f, s in zip(factors, penalty_funcs, penalty_scales)])

    # proximal operator
    _op = prox_nonneg_l1 if nonneg else prox_l1
    prox_ops = [_op if s > 0 else prox_no_penalty for s in penalty_scales]
    loss_hist = []
    obj_hist = []

    for i in range(niter):
        
        factor_cache = [fctr.copy() for fctr in factors]

        for mode in range(tensor.ndim):        
            
            # form unfolding and khatri-rao product
            A = unfold(tensor, mode)
            B = khatri_rao(factors, skip_matrix=mode).T
            X = factors[mode]

            # compute gradient with respect to X
            loss = 0.5*np.mean((A - X.dot(B))**2)
            grad = (1 / np.prod(tensor.shape)) * (X.dot(B).dot(B.T) - A.dot(B.T))
            
            # keep history of loss
            loss_hist.append(loss)
            obj_hist.append(loss + np.sum(l1norms))

            # apply proximal gradient step
            factors[mode] = prox_ops[mode](X - lr*grad, lr, penalties[mode])

            # update 
            penalties[mode] = penalty_funcs[mode](factors[mode], penalty_scales[mode])

        if loss_hist[-1] > loss_hist[-tensor.ndim]:
            lr *= 0.1
            factors = factor_cache
            loss_hist = loss_hist[:-tensor.ndim]
            obj_hist = loss_hist[:-tensor.ndim]
        else:
            lr *= 1.1

        factors = standardize_factors(factors, sort_factors=False)

    return factors, {'loss_hist': loss_hist, 'obj_hist': obj_hist}
