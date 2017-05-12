import tensorly
from tensorly.kruskal import kruskal_to_tensor
from tensorly.base import unfold
from tensorly.tenalg import khatri_rao
from .kruskal import standardize_factors
import numpy as np
from tqdm import trange
from scipy.optimize import check_grad

def nucnorm(matrix, penalty):
    """Calculates the nuclear norm of a matrix
    """
    if penalty <= 0:
        return 0.0
    else:
        s = np.linalg.svd(matrix, compute_uv=False)
        return np.sum(np.abs(s))

def prox_nucnorm(matrix, lr, penalty):
    """Proximal operater on the nuclear norm of a matrix.
    """
    if penalty <= 0:
        return matrix
    else:
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        sthr = np.maximum(s - (penalty * lr), 0)
        return np.dot(u*sthr, v)

def lowrank_cpfit(tensor, rank, penalties, niter=1000, lr=1.0):
    """CP decomposition with a nuclear norm constraint on the factor matrices
    """

    # check inputs
    if not np.iterable(penalties) or len(penalties) != tensor.ndim:
        raise ValueError('Penalties should be specified as a list, matching the number of factor matrices.')
    penalties = np.array(penalties)

    # initialize factors
    factors = [np.random.randn(s, rank) for s in tensor.shape]
    nucnorms = [nucnorm(fctr, p) for fctr, p in zip(factors, penalties)]
    
    loss_hist = []
    obj_hist = []
    nrm = 1 / np.prod(tensor.shape)

    for i in range(niter):

        factor_cache = [fctr.copy() for fctr in factors]

        # alternating optimization over modes
        for mode in range(tensor.ndim):

            # form unfolding and khatri-rao product
            A = unfold(tensor, mode)
            B = khatri_rao(factors, skip_matrix=mode).T
            X = factors[mode]

            # compute gradient with respect to X
            loss = 0.5*np.mean((A - X.dot(B))**2)
            grad = nrm * (X.dot(B).dot(B.T) - A.dot(B.T))
            
            # save loss
            loss_hist.append(loss)
            obj_hist.append(loss + np.sum(nucnorms))

            # # backtracking line search
            # while 0.5*np.sum((A - (X - lr*grad).dot(B))**2) > loss:
            #     lr *= 0.1

            # update params
            factors[mode] = prox_nucnorm(X - lr*grad, lr, penalties[mode])
            nucnorms[mode] = nucnorm(factors[mode], penalties[mode])
        
        if loss_hist[-1] > loss_hist[-tensor.ndim]:
            lr *= 0.1
            factors = factor_cache
        else:
            lr *= 1.1

        # renormalize factors
        factors = standardize_factors(factors, sort_factors=False)

    return factors, {'loss_hist': loss_hist, 'obj_hist': obj_hist}
