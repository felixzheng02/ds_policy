import numpy as np
import cvxpy as cp

from .quat_tools import *
from .plot_tools import *


def optimize_ori(q_in, q_out, q_att, postProb):

    q_in_att    = riem_log(q_att, q_in)

    q_out_body = riem_log(q_in, q_out)            
    q_out_att  = parallel_transport(q_in, q_att, q_out_body)

    K, _ = postProb.shape
    N = 4

    A_vars = []
    constraints = []
    for k in range(K):
        A_vars.append(cp.Variable((N, N), symmetric=False))

        constraints += [A_vars[k]<< np.zeros((4, 4))]
        
        # constraints += [A_vars[k].T + A_vars[k] << np.zeros((4, 4))]
        # constraints += [cp.norm(A_vars[k], 'fro') <= max_norm]
        
    for k in range(K):
        q_pred_k = A_vars[k] @ q_in_att.T
        if k == 0:
            q_pred  = cp.multiply(np.tile(postProb[k, :], (N, 1)), q_pred_k)
        else:
            q_pred += cp.multiply(np.tile(postProb[k, :], (N, 1)), q_pred_k)
    q_pred = q_pred.T

    
    objective = cp.norm(q_out_att-q_pred, 'fro')


    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)


    A_res = np.zeros((K, N, N))
    for k in range(K):
        A_res[k, :, :] = A_vars[k].value

    return A_res