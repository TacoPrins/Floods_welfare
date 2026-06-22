"""
LoM.py
"""
###########################################################
### Imports
import numpy as np
from numba import njit

###########################################################
### Functions
###########################################################
@njit
def LoM(par, grids,t_index,vCoeff):
    t=grids.vTime[t_index]
    t_cheby=(2*t-(grids.vTime[0]+grids.vTime[-1]))/(grids.vTime[-1]-grids.vTime[0])
    dP = vCoeff[0]  
    
    poly_curr_min_2 = 1.0
    poly_curr_min_1 = t_cheby  
    
    if par.order_polynomial >= 1:
        dP += vCoeff[1] *  poly_curr_min_1
    
    for n in range(2, par.order_polynomial + 1):
        poly_curr = 2*t_cheby*poly_curr_min_1 - poly_curr_min_2
        dP += vCoeff[n] * poly_curr
        poly_curr_min_2 = poly_curr_min_1
        poly_curr_min_1 = poly_curr
    
    return dP
    
@njit    
def LoM_path(par, grids, vCoeff):
    """Evaluate Chebyshev price LoM over the full time grid. Returns array."""
    out = np.empty(grids.vTime.size)
    for t_index in range(grids.vTime.size):
        out[t_index] = LoM(par, grids, t_index, vCoeff)
    return out