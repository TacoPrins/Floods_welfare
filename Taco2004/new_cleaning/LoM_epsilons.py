"""
LoM.py
"""
###########################################################
### Imports

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
    
