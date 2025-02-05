import torch
import numpy as np

__ALL__ = ["rand_svd"]


def fast_rsvd(a: torch.Tensor,
            method_args: dict):
    # algorithm 4.4 in Halko's paper
    rank = method_args['rank'] + method_args['over_sample'] 
    min_rank = min((a.shape[0], a.shape[1])) # low_rank svd only allows rank <= min(m, n)
    if rank > min_rank:
        rank = min_rank
    u, s, v = torch.svd_lowrank(a, q=rank, niter=method_args['power'])
    vt = v.t()

    if method_args['record_true_error']:

        a_norm = a.norm(2)
        error_true = (a - u @ torch.diag(s) @ vt).norm(2) / a_norm
        error_approx = error_true
    else:
        error_true = None
        error_approx = None

    return u, s, vt#, error_approx, error_true

def adaptive_rsvd(a: torch.Tensor,
                method_args: dict):
    """
    r: the number of vectors to check
    rank_buffer: the initial rank set for the adaptive method (given by the previous run)
    tolerance: the RELATIVE tolerance measured in l2 norm, should be like 0.1, etc
    max_rank: the maximum rank of the matrix
    max_iter: the maximum iteration
    step_size: the step size for searching
    record_true_error: whether to record the true error. used when debugging. this will take more time to compute
    """

    m, n = a.shape
    if m < n or n > method_args['rank']:
        # transpose on a so that the matrix to be svd is small (the m<n) case or the v matrix shape is small (the n>init_rank) case
        Q, error_approx, error_true = adaptive_randomized_range_finder(A = a.t(), 
                                            method_args = method_args)

        B_t = torch.matmul(a, Q)
        u, s, vt = torch.linalg.svd(B_t, full_matrices=False)
        vt = vt @ Q.t()
    else:
        # m > n and n <= init_rank
        Q, error_approx, error_true = adaptive_randomized_range_finder(A = a, 
                                            method_args = method_args)
        B = torch.matmul(Q.t(), a)
        u, s, vt = torch.linalg.svd(B, full_matrices=False)
        u = Q @ u

    method_args['rank'] = Q.shape[1] # update the rank

    return u, s, vt#, error_approx, error_true

def adaptive_randomized_range_finder(
    A: torch.Tensor,
    method_args: dict,
):
    """
    Code modified from https://www.lyndonduong.com/rangefinder/

    Adaptive range finder Algo 4.2 from Halko, Martinsson, Tropp 2011.
    Given an m × n matrix A, a tolerance epsilon, and an integer r
    (e.g. r = 10), this scheme computes an orthonormal matrix Q s.t.
    |(I-Q*Q.T)*A|<= epsilon holds w/ probability
    at least 1 − min{m, n} 10^−r.

    A: input matrix
    r: the number of vectors to check
    rel_tolerance: the relative tolerance, should be like 0.1, etc
    max_rank: the maximum rank of the matrix
    max_iter: the maximum iteration
    init_rank: the initial rank set for the adaptive method (given by the previous run)
    """

    m, n = A.shape
    A_norm = A.norm(p=2)
    # the threshold is related to our tolerance divided by a fixed constant
    if method_args['with_factor']:
        factor = 10*np.sqrt(2/np.pi)
    else:
        factor = 1.
    rel_thresh = method_args['tolerance'] / factor

    max_iter = method_args['max_rank']

    I = torch.eye(m, device = A.device)
    error_true = None

    # initial range space estimate (single vector)
    if method_args['rank'] > 0:
        # initialize Q with predefined rank
        omega = torch.randn(n, method_args['rank'], device=A.device)
        y = A @ omega
        Q, _ = torch.linalg.qr(y) # the column space determined by the initial rank

        # test the approximation error
        omega = torch.randn(n, method_args['r'], device=A.device)
        y = (I - Q @ Q.T) @ (A @ omega)
        error_approx = y.norm(dim=0, p=2).max() / A_norm
        if method_args['record_true_error']:
            error_true = ((I - Q @ Q.T) @ A).norm(p=2) / A_norm

        if Q.shape[1] >= method_args['max_rank'] or error_approx <= rel_thresh:
            # if the rank is already large enough or the error is small enough, return
            return Q, error_approx*factor, error_true

    else:
        # initialize empty Q
        Q = torch.empty(m, 0)
        if method_args['step_size'] == 1:
            # draw some new y here
            omega = torch.randn(n, method_args['r'], device=A.device)           # draw standard gaussian vecs
            y = A @ omega # (m,r)
        error_approx = 1e5

    for j in range(max_iter):

        if  error_approx <= rel_thresh or Q.shape[1] >= method_args['max_rank']:
            break
        else:
            if method_args['step_size'] == 1:
                # follow the exact algorithm 4.2
                # overwrite y_j : 
                # reproject onto the range orthog to Q, i.e. (y_j - Q*Q.T*y_j)
                y[:, j] = (I - Q@Q.T) @ y[:, j]

                # normalize it and concate to Q
                q = y[:,j] / y[:,j].norm(p=2)
                Q = torch.cat([Q, q.unsqueeze(-1)], -1)

                # draw new gaussian vec
                omega = torch.randn(n, 1, device = A.device)

                # get approximation error = (A @ w - QQ.T @ w)
                y_add = (I - Q @ Q.T) @ (A @ omega)

                # append to y's
                y = torch.cat([y, y_add], -1)  

                # overwrite j+1:j+r-1 vecs
                y[:, j+1:j+r] = (I - torch.outer(q,q)) @ y[:, j+1:j+r]

                # compute error of last r consecutive vecs
                error_approx = y[:, -method_args['r']:].norm(dim=0, p=2).max() / A_norm
            else:
                # update by step_size.
                omega = torch.randn(n, method_args['step_size'], device=A.device)
                y = (I - Q @ Q.T) @ (A @ omega)
                Q = torch.cat([Q, y], -1)
                # do qr
                Q, _ = torch.linalg.qr(Q, mode='reduced')
                # compute the error
                omega = torch.randn(n, method_args['r'], device=A.device)
                y = (I - Q @ Q.T) @ (A @ omega)
                error_approx = y.norm(dim=0, p=2).max() / A_norm

    # compute true error
    if method_args['record_true_error']:
        error_true = ((I - Q @ Q.T) @ A).norm(p=2) / A_norm

    return Q, error_approx * factor, error_true
