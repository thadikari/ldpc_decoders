import numpy as np
import itertools
from scipy.optimize import linprog
import math_utils as mu


class LP:
    id_keys = ['max_iter', 'allow_pseudo']

    def __init__(self, parity_mtx, **kwargs):
        self.allow_pseudo = kwargs['allow_pseudo']
        self.max_iter = kwargs['max_iter']
        num_chk, num_var = parity_mtx.shape
        num_constraints = np.sum(2 ** (parity_mtx.sum(axis=1) - 1))
        self.mat_ub = np.zeros((num_constraints, num_var), int)
        self.b_ub = np.zeros(num_constraints, int)

        cs = 0
        for chk_ind in range(num_chk):
            chk_yy = np.where(parity_mtx[chk_ind])[0]
            chk_deg = chk_yy.shape[0]
            str_seq = [seq for seq in itertools.product("01", repeat=chk_deg)]
            all_sets = np.array(str_seq).astype(np.int)
            sums = all_sets.sum(axis=1)
            idx = (sums % 2) == 1
            odd_sets = all_sets[idx, :]
            alloc = odd_sets.shape[0]
            self.mat_ub[cs:cs + alloc, chk_yy] = odd_sets * 2 - 1
            self.b_ub[cs:cs + alloc] = sums[idx] - 1
            cs += alloc

    def decode(self, y, gamma):
        res = linprog(gamma, A_ub=self.mat_ub,
                      b_ub=self.b_ub, bounds=(0, 1),
                      # options={"disp": True, "maxiter": self.max_iter}
                      )
        return mu.pseudo_to_cw(res.x, self.allow_pseudo)
