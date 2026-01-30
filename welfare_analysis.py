# -*- coding: utf-8 -*-
import numpy as np
import misc_functions as misc
import household_problem_epsilons_nolearning as household_problem  
import simulation as sim
from numba import njit

@njit
def welfare_calculator_initial(method, par, grids, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_initial,vCoeff_NC_initial, max_ltv):
    
    #Begin with initial steady state with constant flood risk
   
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_initial(grids, par, dPi_L, iNj, mMarkov, vCoeff_C_initial[0],vCoeff_NC_initial[0])
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0 = sim.stat_dist_finder(False, grids, par, mMarkov, dPi_L, iNj, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vCoeff_C_initial,vCoeff_NC_initial)
    
    vt_stay_c_expanded=grid_adjust_initial(grids,iNj,vt_stay_c)
    vt_stay_nc_expanded=grid_adjust_initial(grids,iNj,vt_stay_nc)
    vt_renter_expanded=grid_adjust_initial_rentshape(grids,iNj,vt_renter)

    #initial_welfare=np.sum(mDist1_renter[:,:,:,:,:]*vt_renter_expanded[0,:,:,:,0,:,:])+np.sum(mDist1_c[:,:,:,:,:,:,:]*vt_stay_c_expanded[0,:,:,:,0,:,:,:,:])+np.sum(mDist1_nc[:,:,:,:,:,:,:]*vt_stay_nc_expanded[0,:,:,:,0,:,:,:,:])
    initial_welfare=0
    initial_welfare_vec=np.zeros((grids.vX_sim.size))
    for t_index in range(9, grids.vTime.size):
        #initial_welfare+=(par.dBeta)**(par.time_increment*t_index)*(np.sum(mDist1_renter[0,:,:,:,:]*vt_renter_expanded[0,0,:,:,0,:,:]))
        initial_welfare+=(np.sum(mDist1_renter[0,:,:,:,:]*vt_renter_expanded[0,:,:,:,:]))
        for x_index_sim in range(grids.vX_sim.size):
            initial_welfare_vec[x_index_sim]+=(np.sum(mDist1_renter[0,:,:,x_index_sim,:]*vt_renter_expanded[0,:,:,x_index_sim,:]))
    return initial_welfare, initial_welfare_vec
@njit
def welfare_calculator_re(method, par, grids, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_initial,vCoeff_NC_initial, max_ltv):
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_initial(grids, par, dPi_L, iNj, mMarkov, vCoeff_C_initial[0],vCoeff_NC_initial[0])
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0 = sim.stat_dist_finder(False, grids, par, mMarkov, dPi_L, iNj, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vCoeff_C_initial,vCoeff_NC_initial)

    
    vt_stay_c, vt_stay_nc, vt_renter,_, _, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf, _, _=household_problem.solve_welfare(grids, par, dPi_L, iNj, mMarkov,vCoeff_C_RE,vCoeff_NC_RE)  
        
    vt_stay_c_expanded=grid_adjust(grids,iNj,vt_stay_c)
    vt_stay_nc_expanded=grid_adjust(grids,iNj,vt_stay_nc)
    vt_renter_expanded=grid_adjust_rentshape(grids,iNj,vt_renter)
    
    vt_stay_c_wf_expanded=grid_adjust(grids,iNj,vt_stay_c_wf)
    vt_stay_nc_wf_expanded=grid_adjust(grids,iNj,vt_stay_nc_wf)
    vt_renter_wf_expanded=grid_adjust_rentshape(grids,iNj,vt_renter_wf)
                  
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock0, coastal_beq, noncoastal_beq, savings_beq = sim.stat_dist_finder(False, grids, par, mMarkov, dPi_L, iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_RE,vCoeff_NC_RE)

    #re_welfare=np.sum(mDist1_renter[:,0,:,:,:]*vt_renter_expanded[0,:,0,:,0,:,:])+np.sum(mDist1_c[:,0,:,:,:,:,:]*vt_stay_c_expanded[0,:,0,:,0,:,:,:,:])+np.sum(mDist1_nc[:,0,:,:,:,:,:]*vt_stay_nc_expanded[0,:,0,:,0,:,:,:,:])
    re_welfare=0
    re_welfare_vec=np.zeros((grids.vX_sim.size))
    max_x_value=2/3*grids.vX_sim[-1]
    mDist1_renter[0,:,:,:,:]= sim.initial_dist_0(grids, mMarkov, iNj, max_x_value, False)
    ##INITIAL DISTRIBUTION WILL BECOME TIME-DEPENDENT EVENTUALLY!!
    for t_index in range(9, grids.vTime.size):
        #re_welfare+=(par.dBeta)**(par.time_increment*t_index)*np.sum(mDist1_renter[0,:,:,:,:]*vt_renter_expanded[t_index,0,:,:,0,:,:])
        re_welfare+=np.sum(mDist1_renter[0,:,:,:,:]*vt_renter_expanded[t_index,0,:,:,:,:])
        for x_index_sim in range(grids.vX_sim.size):
            re_welfare_vec[x_index_sim]+=np.sum(mDist1_renter[0,:,:,x_index_sim,:]*vt_renter_expanded[t_index,0,:,:,x_index_sim,:])
    #vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_initial(grids, par, dPi_L, iNj, mMarkov, vCoeff_in_C_initial[0],vCoeff_in_NC_initial[0])
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0 = sim.stat_dist_finder(True, grids, par, mMarkov, dPi_L, iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_in_C_initial,vCoeff_in_NC_initial)
    mDist1_renter[0,:,:,:,:]= sim.initial_dist_0(grids, mMarkov, iNj, max_x_value, True)
    partial_welfare=np.zeros((2))
    partial_welfare_mat=np.zeros((2, grids.vX_sim.size))
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock0, coastal_beq, noncoastal_beq, savings_beq = sim.stat_dist_finder(True, grids, par, mMarkov, dPi_L, iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_RE,vCoeff_NC_RE)
    #partial_welfare[0]=np.sum(mDist1_renter[:,0,:,:,:]*vt_renter_expanded[0,:,0,:,0,:,:])+np.sum(mDist1_c[:,0,:,:,:,:,:]*vt_stay_c_expanded[0,:,0,:,0,:,:,:,:])+np.sum(mDist1_nc[:,0,:,:,:,:,:]*vt_stay_nc_expanded[0,:,0,:,0,:,:,:,:])
    #partial_welfare[1]=np.sum(mDist1_renter[:,1,:,:,:]*vt_renter_wf_expanded[0,:,1,:,0,:,:])+np.sum(mDist1_c[:,1,:,:,:,:,:]*vt_stay_c_wf_expanded[0,:,1,:,0,:,:,:,:])+np.sum(mDist1_nc[:,1,:,:,:,:,:]*vt_stay_nc_wf_expanded[0,:,1,:,0,:,:,:,:])
    for t_index in range(9, grids.vTime.size):
        #partial_welfare[0]+=(par.dBeta)**(par.time_increment*t_index)*np.sum(mDist1_renter[0,0,:,:,:]*vt_renter_expanded[t_index,0,0,:,0,:,:])
        #partial_welfare[1]+=(par.dBeta)**(par.time_increment*t_index)*np.sum(mDist1_renter[0,1,:,:,:]*vt_renter_wf_expanded[t_index,0,1,:,0,:,:])
        partial_welfare[0]+=np.sum(mDist1_renter[0,0,:,:,:]*vt_renter_expanded[t_index,0,0,:,:,:])
        partial_welfare[1]+=np.sum(mDist1_renter[0,1,:,:,:]*vt_renter_wf_expanded[t_index,0,1,:,:,:])
        for x_index_sim in range(grids.vX_sim.size):
            partial_welfare_mat[0, x_index_sim]+=np.sum(mDist1_renter[0,0,:,x_index_sim,:]*vt_renter_expanded[t_index,0,0,:,x_index_sim,:])
            partial_welfare_mat[1, x_index_sim]+=np.sum(mDist1_renter[0,1,:,x_index_sim,:]*vt_renter_wf_expanded[t_index,0,1,:,x_index_sim,:])
    return re_welfare, partial_welfare, re_welfare_vec, partial_welfare_mat
@njit      
def welfare_calculator_he(method, par, grids, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_initial,vCoeff_NC_initial, max_ltv):
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_initial(grids, par, dPi_L, iNj, mMarkov, vCoeff_C_initial[0],vCoeff_NC_initial[0])
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0 = sim.stat_dist_finder(True, grids, par, mMarkov, dPi_L, iNj, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vCoeff_C_initial,vCoeff_NC_initial)

    vt_stay_c, vt_stay_nc, vt_renter,_, _, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf, _, _=household_problem.solve_welfare(grids, par, dPi_L, iNj, mMarkov,vCoeff_C,vCoeff_NC)  
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock0, coastal_beq, noncoastal_beq, savings_beq = sim.stat_dist_finder(True, grids, par, mMarkov, dPi_L, iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C,vCoeff_NC)
    
    vt_stay_c_expanded=grid_adjust(grids,iNj,vt_stay_c)
    vt_stay_nc_expanded=grid_adjust(grids,iNj,vt_stay_nc)
    vt_renter_expanded=grid_adjust_rentshape(grids,iNj,vt_renter)
    
    vt_stay_c_wf_expanded=grid_adjust(grids,iNj,vt_stay_c_wf)
    vt_stay_nc_wf_expanded=grid_adjust(grids,iNj,vt_stay_nc_wf)
    vt_renter_wf_expanded=grid_adjust_rentshape(grids,iNj,vt_renter_wf)
    
    he_welfare=np.zeros((2))  
    he_welfare_mat=np.zeros((2, grids.vX_sim.size))  
    #he_welfare[0]=np.sum(mDist1_renter[:,0,:,:,:]*vt_renter_expanded[0,:,0,:,0,:,:])+np.sum(mDist1_c[:,0,:,:,:,:,:]*vt_stay_c_expanded[0,:,0,:,0,:,:,:,:])+np.sum(mDist1_nc[:,0,:,:,:,:,:]*vt_stay_nc_expanded[0,:,0,:,0,:,:,:,:])
    #he_welfare[1]=np.sum(mDist1_renter[:,1,:,:,:]*vt_renter_wf_expanded[0,:,1,:,0,:,:])+np.sum(mDist1_c[:,1,:,:,:,:,:]*vt_stay_c_wf_expanded[0,:,1,:,0,:,:,:,:])+np.sum(mDist1_nc[:,1,:,:,:,:,:]*vt_stay_nc_wf_expanded[0,:,1,:,0,:,:,:,:])
    max_x_value=2/3*grids.vX_sim[-1]
    mDist1_renter[0,:,:,:,:]= sim.initial_dist_0(grids, mMarkov, iNj, max_x_value, True)
    for t_index in range(9, grids.vTime.size):        
        #he_welfare[0]+=(par.dBeta)**(par.time_increment*t_index)*np.sum(mDist1_renter[0,0,:,:,:]*vt_renter_expanded[t_index,0,0,:,0,:,:])
        #he_welfare[1]+=(par.dBeta)**(par.time_increment*t_index)*np.sum(mDist1_renter[0,1,:,:,:]*vt_renter_wf_expanded[t_index,0,1,:,0,:,:])
        he_welfare[0]+=np.sum(mDist1_renter[0,0,:,:,:]*vt_renter_expanded[t_index,0,0,:,:,:])
        he_welfare[1]+=np.sum(mDist1_renter[0,1,:,:,:]*vt_renter_wf_expanded[t_index,0,1,:,:,:])
        for x_index_sim in range(grids.vX_sim.size):
            he_welfare_mat[0, x_index_sim]+=np.sum(mDist1_renter[0,0,:,x_index_sim,:]*vt_renter_expanded[t_index,0,0,:,x_index_sim,:])
            he_welfare_mat[1, x_index_sim]+=np.sum(mDist1_renter[0,1,:,x_index_sim,:]*vt_renter_wf_expanded[t_index,0,1,:,x_index_sim,:])
    return he_welfare, he_welfare_mat

@njit
def compute_p_left(grid, x, i_left):
    
    x_left = grid[i_left]
    x_right = grid[i_left + 1]
    p_left = (x_right - x) / (x_right - x_left)

    return p_left


@njit 
def grid_adjust(grids,iNj,vt_stay_in):
    vt_stay_expanded=np.zeros((grids.vTime.size, iNj, grids.vK.size, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size, grids.vE.size))

    for m_index_sim in range(grids.vM_sim.size):
        m=grids.vM_sim[m_index_sim]
        m_index_l=misc.binary_search(0, grids.vM.size, grids.vM,m) 
        m_weight_left=compute_p_left(grids.vM, m, m_index_l)
        for l_index_sim in range(grids.vL_sim.size):
            ltv=grids.vL_sim[l_index_sim]
            l_index_l=misc.binary_search(0, grids.vL.size, grids.vL,ltv) 
            l_weight_left=compute_p_left(grids.vL, ltv, l_index_l)
            vt_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*l_weight_left*vt_stay_in[:,:,:,:,m_index_l,:,l_index_l,:]
            vt_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*l_weight_left*vt_stay_in[:,:,:,:,m_index_l+1,:,l_index_l,:]
            vt_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*(1-l_weight_left)*vt_stay_in[:,:,:,:,m_index_l,:,l_index_l+1,:]
            vt_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*(1-l_weight_left)*vt_stay_in[:,:,:,:,m_index_l+1,:,l_index_l+1,:] 
    return vt_stay_expanded

@njit 
def grid_adjust_rentshape(grids,iNj,vt_renter_in):
    vt_rentshape_expanded=np.zeros((grids.vTime.size, iNj, grids.vK.size, grids.vG.size, grids.vX_sim.size, grids.vE.size))
    for x_index_sim in range(grids.vX_sim.size):
        x=grids.vX_sim[x_index_sim]
        x_index_l=misc.binary_search(0, grids.vX.size, grids.vX,x) 
        x_weight_left=compute_p_left(grids.vX, x, x_index_l)
        vt_rentshape_expanded[:,:,:,:,x_index_sim,:]+=x_weight_left*vt_renter_in[:,:,:,:,x_index_l,:]
        vt_rentshape_expanded[:,:,:,:,x_index_sim,:]+=(1-x_weight_left)*vt_renter_in[:,:,:,:,x_index_l+1,:]  
    return vt_rentshape_expanded

@njit 
def grid_adjust_initial(grids,iNj,vt_stay_in):
    vt_stay_expanded=np.zeros((iNj, grids.vK.size, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size, grids.vE.size))

    for m_index_sim in range(grids.vM_sim.size):
        m=grids.vM_sim[m_index_sim]
        m_index_l=misc.binary_search(0, grids.vM.size, grids.vM,m) 
        m_weight_left=compute_p_left(grids.vM, m, m_index_l)
        for l_index_sim in range(grids.vL_sim.size):
            ltv=grids.vL_sim[l_index_sim]
            l_index_l=misc.binary_search(0, grids.vL.size, grids.vL,ltv) 
            l_weight_left=compute_p_left(grids.vL, ltv, l_index_l)
            vt_stay_expanded[:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*l_weight_left*vt_stay_in[:,:,:,m_index_l,:,l_index_l,:]
            vt_stay_expanded[:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*l_weight_left*vt_stay_in[:,:,:,m_index_l+1,:,l_index_l,:]
            vt_stay_expanded[:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*(1-l_weight_left)*vt_stay_in[:,:,:,m_index_l,:,l_index_l+1,:]
            vt_stay_expanded[:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*(1-l_weight_left)*vt_stay_in[:,:,:,m_index_l+1,:,l_index_l+1,:] 
    return vt_stay_expanded

@njit 
def grid_adjust_initial_rentshape(grids,iNj,vt_renter_in):
    vt_rentshape_expanded=np.zeros((iNj, grids.vK.size, grids.vG.size, grids.vX_sim.size, grids.vE.size))
    for x_index_sim in range(grids.vX_sim.size):
        x=grids.vX_sim[x_index_sim]
        x_index_l=misc.binary_search(0, grids.vX.size, grids.vX,x) 
        x_weight_left=compute_p_left(grids.vX, x, x_index_l)
        vt_rentshape_expanded[:,:,:,x_index_sim,:]+=x_weight_left*vt_renter_in[:,:,:,x_index_l,:]
        vt_rentshape_expanded[:,:,:,x_index_sim,:]+=(1-x_weight_left)*vt_renter_in[:,:,:,x_index_l+1,:]  
    return vt_rentshape_expanded