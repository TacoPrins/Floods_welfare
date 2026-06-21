# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:00:10 2026

@author: tprins
"""

import numpy as np
import misc_functions as misc
import household_problem_epsilons_nolearning as household_problem  
import simulation as sim
import equilibrium as equil
import experiments as experiments
import simulate_initial_joint as initial_joint_sim
from numba import njit

@njit
def find_expenditure_equiv_EK_SLR(par, grids, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C_in, vCoeff_NC_in, config):


    initial = True
    welfare = False
    stop_2026 = False
    # run and save SS without welfare: get stationary dist
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve_ss(grids, par,vCoeff_C_initial[0], vCoeff_NC_initial[0], config)
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, _=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, config)
    
    # get value functions over transition with SLR
    welfare = True
    if config.sceptics == False:
        k_dim=1
    else:
        k_dim=grids.vK.size        
    

    price_history, _, _, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf,_,_,_=equil.generate_pricepath(grids, par, vCoeff_C_in,vCoeff_NC_in, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, sceptics, stop_2026, welfare)
    v_nonowner_wf_expanded_SLR=grid_adjust_rentshape(par,grids,v_nonowner_wf)
    v_owner_c_wf_expanded_SLR=grid_adjust(par,grids,v_owner_c_wf)
    v_owner_nc_wf_expanded_SLR=grid_adjust(par,grids,v_owner_nc_wf)
    
    
    wf_loss = np.linspace(-0.10, 0.15,100)
    
    ce_C  = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_NC = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter_newborns = np.zeros((wf_loss.size,grids.vTime.size, k_dim, grids.vE.size))
    
    wf_SLR_c=np.zeros((k_dim,grids.vE.size))
    wf_SLR_nc=np.zeros((k_dim,grids.vE.size))
    wf_SLR_rent=np.zeros((k_dim,grids.vE.size))
    wf_SS_newborns=np.zeros((wf_loss.size,k_dim,grids.vE.size))
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):
            wf_SLR_c[k_index,e_index] = np.sum(mDist1_c_SS[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_SLR[0,1:,k_index, :, :,:,:,e_index])
            wf_SLR_nc[k_index,e_index] = np.sum(mDist1_nc_SS[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_SLR[0,1:,k_index, :, :,:,:,e_index])
            wf_SLR_rent[k_index,e_index] = np.sum(mDist1_renter_SS[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_SLR[0,:,k_index, :, :,e_index])
    
    
    for wf_idx in range(wf_loss.size):
        par.wf_wedge[0] = wf_loss[wf_idx]
        print(par.wf_wedge[0])

        v_owner_c_wf_SS, v_owner_nc_wf_SS, v_nonowner_wf_SS, _, _, _ = household_problem.solve_ss(grids, par, vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
        v_nonowner_wf_expanded_SS=grid_adjust_rentshape(par,grids,v_nonowner_wf_SS)
        v_owner_c_wf_expanded_SS=grid_adjust(par,grids,v_owner_c_wf_SS)
        v_owner_nc_wf_expanded_SS=grid_adjust(par,grids,v_owner_nc_wf_SS)
        
        
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                wf_SS_c = np.sum(mDist1_c_SS[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_SS[0,1:,k_index, :, :,:,:,e_index])
                wf_SS_nc = np.sum(mDist1_nc_SS[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_SS[0,1:,k_index, :, :,:,:,e_index])
                wf_SS_rent = np.sum(mDist1_renter_SS[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_SS[0,:,k_index, :, :,e_index])
                
                ce_C[wf_idx,k_index, e_index] = wf_SS_c - wf_SLR_c[k_index,e_index]
                ce_NC[wf_idx,k_index, e_index] = wf_SS_nc - wf_SLR_nc[k_index,e_index]
                ce_renter[wf_idx,k_index, e_index] = wf_SS_rent - wf_SLR_rent[k_index,e_index]
        
 
        
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                wf_SS_newborns[wf_idx, k_index,e_index] = np.sum(mDist1_renter_SS[0,k_index, :, :,e_index]* v_nonowner_wf_expanded_SS[0,0,k_index, :, :, e_index])
            
        
    for t_index in range(grids.vTime.size):
        dP_C=price_history[t_index,0]
        dP_NC=price_history[t_index,1]
        coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
        housing_bequest=coastal_beq*(1-coastal_damage_frac-par.dDelta)*dP_C + noncoastal_beq*(1-par.dDelta)*dP_NC
        total_bequest = (housing_bequest+savings_beq*(1+par.r))*par.iNj
        mPi_joint=initial_joint_sim.initial_joint(par, grids, total_bequest)
        # weight
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                if sceptics==True:
                    mDist1_renter[0,k_index,:,:,e_index]= (1/par.iNj)*(1/grids.vG.size)*grids.vTypes[k_index]*mPi_joint[:,e_index]        
                else:
                    mDist1_renter[0,k_index,:,:,e_index]= (1/par.iNj)*(1/grids.vG.size)*mPi_joint[:,e_index]     
                wf_SLR_newborns = np.sum(mDist1_renter[0,k_index, :, :,e_index]* v_nonowner_wf_expanded_SLR[t_index,0,k_index, :, :,e_index])
                for wf_idx in range(wf_loss.size):
                    ce_renter_newborns[wf_idx,t_index, k_index, e_index] = wf_SS_newborns[wf_idx, k_index,e_index] - wf_SLR_newborns 

        if t_index<grids.vTime.size-1:
            coastal_beq=vcoastal_beq[t_index]
            noncoastal_beq=vnoncoastal_beq[t_index]
            savings_beq=vsavings_beq[t_index]
            
                        
    par.wf_wedge[0] = 0.0
    
    
    tax_equiv_C = np.zeros((k_dim, grids.vE.size))
    tax_equiv_NC = np.zeros((k_dim, grids.vE.size))
    tax_equiv_renter = np.zeros((k_dim, grids.vE.size))
    tax_equiv_newborns = np.zeros((grids.vTime.size, k_dim, grids.vE.size))
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):            
            
            tax_equiv_C[k_index, e_index] = find_zero_linear(wf_loss, ce_C[:,k_index, e_index]) # find wf_loss that makes ce_C[:,k_index, g_index, e_index] equal to zero
            tax_equiv_NC[k_index, e_index] = find_zero_linear(wf_loss, ce_NC[:,k_index, e_index])  # find wf_loss that makes ce_NC[:,k_index, g_index, e_index] equal to zero
            tax_equiv_renter[k_index, e_index] = find_zero_linear(wf_loss, ce_renter[:,k_index, e_index])  # find wf_loss that makes ce_renter[:,k_index, g_index, e_index] equal to zero

            for t_index in range(grids.vTime.size):
                tax_equiv_newborns[t_index, k_index, e_index] = find_zero_linear(wf_loss, ce_renter_newborns[:,t_index, k_index, e_index])
                    

    return tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns


@njit
def find_expenditure_equiv_EK_policy(par, grids, grids_pol, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC, vCoeff_C_pol, vCoeff_NC_pol, sceptics=True, building_rest = False, mortgage_premium = False):
    """
    New: import the value functions at the time of policy introduction as well as the distribution:
    I call it the 'business as usual' (BAU) value functions
        
    v_nonowner_wf_BAU: shape 1 x J x K x G x M x E (value functions from 2026 onwards)
    v_owner_c_wf_BAU: shape 1 x J x K x G x H x L x  M x E
    v_owner_nc_wf_BAU: shape 1 x J x K x G x  H x L x M x E
    dist_c_BAU: J x K x G x M x H x L x E (2026 distribution)
    dist_nc_BAU: J x K x G x M x H x L x E 
    dist_renter_BAU: J x K x G x M x E 
    
    Import everything that would have been the case with business as usual. Solve for the policy case + some 
    lumpsum expenditure tax that makes the value function in the policy case equivalent to the BAU case.
    """
    
    if sceptics == False:
        k_dim=1
    else:
        k_dim=grids.vK.size        


    welfare = True

    price_history_before_pol, mDist0_c, mDist0_nc, mDist0_renter, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq_before, vnoncoastal_beq_before, vsavings_beq_before=experiments.gen_distribution_now(grids, par, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial)
    # get value functions over transition with experiment
    stop_2026 = False
    coastal_beq=vcoastal_beq_before[-1]
    noncoastal_beq=vnoncoastal_beq_before[-1]
    savings_beq=vsavings_beq_before[-1]
    
    dP_C_initial=price_history_before_pol[0,-2]
    dP_NC_initial=price_history_before_pol[1,-2]
    

    plot_stocks=False
    price_history_after_pol, _, _, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq_pol, vnoncoastal_beq_pol, vsavings_beq_pol, _, _, _, v_owner_c_wf_pol, v_owner_nc_wf_pol, v_nonowner_wf_pol,_,_,_=equil.generate_pricepath(grids_pol, par, vCoeff_C_pol,vCoeff_NC_pol, dP_C_initial, dP_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, stock_demand_rental_C, stock_demand_rental_NC, coastal_beq, noncoastal_beq, savings_beq, sceptics, stop_2026, welfare, plot_stocks, building_rest, mortgage_premium)
    building_rest=False
    mortgage_premium = False
    
    ##WORK IN PROGRESS - THIS FUNCTION DOES NOT WORK AS EXPECTED YET
    vCoeff_C_BAU=retransform_coeffs(vCoeff_C, grids.vTime, grids_pol.vTime)
    vCoeff_NC_BAU=retransform_coeffs(vCoeff_NC, grids.vTime, grids_pol.vTime)
    price_history_after_BAU, _, _, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq_BAU, vnoncoastal_beq_BAU, vsavings_beq_BAU, _, _, _, v_owner_c_wf_BAU, v_owner_nc_wf_BAU, v_nonowner_wf_BAU,_,_,_=equil.generate_pricepath(grids_pol, par, vCoeff_C_BAU,vCoeff_NC_BAU, dP_C_initial, dP_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, stock_demand_rental_C, stock_demand_rental_NC, coastal_beq, noncoastal_beq, savings_beq, sceptics, stop_2026, welfare, plot_stocks, building_rest, mortgage_premium)


    v_owner_c_wf_expanded_pol=grid_adjust(par,grids,v_owner_c_wf_pol)
    v_owner_nc_wf_expanded_pol=grid_adjust(par,grids,v_owner_nc_wf_pol)
    v_nonowner_wf_expanded_pol=grid_adjust_rentshape(par,grids,v_nonowner_wf_pol)
    
    
    wf_loss = np.linspace(-0.10, 0.15,100)
    
    ce_C  = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_NC = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter_newborns = np.zeros((wf_loss.size,grids.vTime.size, k_dim, grids.vE.size))
    
    wf_pol_c=np.zeros((k_dim,grids.vE.size))
    wf_pol_nc=np.zeros((k_dim,grids.vE.size))
    wf_pol_rent=np.zeros((k_dim,grids.vE.size))
    wf_pol_newborns=np.zeros((grids_pol.vTime,k_dim,grids.vE.size))       
   
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):
            wf_pol_c[k_index,e_index] = np.sum(mDist0_c[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_pol[0,1:,k_index, :, :,:,:,e_index])
            wf_pol_nc[k_index,e_index] = np.sum(mDist0_nc[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_pol[0,1:,k_index, :, :,:,:,e_index])
            wf_pol_rent[k_index,e_index] = np.sum(mDist0_renter[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_pol[0,:,k_index, :, :,e_index])
            
            
            
    coastal_beq_pol=coastal_beq
    noncoastal_beq_pol=noncoastal_beq
    savings_beq_pol=savings_beq
    
    for t_index in range(grids_pol.vTime.size):
        dP_C=price_history_after_pol[t_index,0]
        dP_NC=price_history_after_pol[t_index,1]
        coastal_damage_frac=grids_pol.vPi_S_median[t_index]*np.dot(grids_pol.vPDF_z[1:],(1-grids_pol.vZ[1:]))
        housing_bequest=coastal_beq_pol*(1-coastal_damage_frac-par.dDelta)*dP_C + noncoastal_beq_pol*(1-par.dDelta)*dP_NC
        total_bequest = (housing_bequest+savings_beq_pol*(1+par.r))*par.iNj
        mPi_joint=initial_joint_sim.initial_joint(par, grids_pol, total_bequest)
        # weight
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                if sceptics==True:
                    mDist1_renter[0,k_index,:,:,e_index]= (1/par.iNj)*(1/grids.vG.size)*grids.vTypes[k_index]*mPi_joint[:,e_index]        
                else:
                    mDist1_renter[0,k_index,:,:,e_index]= (1/par.iNj)*(1/grids.vG.size)*mPi_joint[:,e_index]     
                wf_pol_newborns[t_index,k_index,e_index] = np.sum(mDist1_renter[0,k_index, :, :,e_index]* v_nonowner_wf_expanded_pol[t_index,0,k_index, :, :,e_index])
        if t_index<grids.vTime.size-1:
            coastal_beq_pol=vcoastal_beq_pol[t_index]
            noncoastal_beq_pol=vnoncoastal_beq_pol[t_index]
            savings_beq_pol=vsavings_beq_pol[t_index]
     

    for wf_idx in range(wf_loss.size):
        par.wf_wedge[0] = wf_loss[wf_idx]
        print(par.wf_wedge[0])
        _, _, _, _, _, _, v_owner_c_wf_BAU, v_owner_nc_wf_BAU, v_nonowner_wf_BAU = household_problem.solve(grids_pol, par, vCoeff_C,vCoeff_NC, sceptics, welfare, building_rest = building_rest, mortgage_premium=mortgage_premium)
        v_owner_c_wf_expanded_BAU=grid_adjust(par,grids,v_owner_c_wf_BAU)
        v_owner_nc_wf_expanded_BAU=grid_adjust(par,grids,v_owner_nc_wf_BAU)
        v_nonowner_wf_expanded_BAU=grid_adjust_rentshape(par,grids,v_nonowner_wf_BAU)
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                ce_C[wf_idx,k_index,e_index]=np.sum(mDist0_c[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_BAU[0,1:,k_index, :, :,:,:,e_index])-wf_pol_c[k_index,e_index]
                ce_NC[wf_idx,k_index,e_index]=np.sum(mDist0_nc[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_BAU[0,1:,k_index, :, :,:,:,e_index])-wf_pol_nc[k_index,e_index]
                ce_renter[wf_idx,k_index,e_index]=np.sum(mDist0_renter[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_BAU[0,:,k_index, :, :,e_index])-wf_pol_rent[k_index,e_index]
        coastal_beq_BAU=coastal_beq
        noncoastal_beq_BAU=noncoastal_beq
        savings_beq_BAU=savings_beq
        
        for t_index in range(grids_pol.vTime.size):
            dP_C=price_history_after_BAU[t_index,0]
            dP_NC=price_history_after_BAU[t_index,1]
            coastal_damage_frac=grids_pol.vPi_S_median[t_index]*np.dot(grids_pol.vPDF_z[1:],(1-grids_pol.vZ[1:]))
            housing_bequest=coastal_beq_BAU*(1-coastal_damage_frac-par.dDelta)*dP_C + noncoastal_beq_BAU*(1-par.dDelta)*dP_NC
            total_bequest = (housing_bequest+savings_beq_BAU*(1+par.r))*par.iNj
            mPi_joint=initial_joint_sim.initial_joint(par, grids_pol, total_bequest)
            # weight
            for k_index in range(k_dim):
                for e_index in range(grids.vE.size):
                    if sceptics==True:
                        mDist1_renter[0,k_index,:,:,e_index]= (1/par.iNj)*(1/grids.vG.size)*grids.vTypes[k_index]*mPi_joint[:,e_index]        
                    else:
                        mDist1_renter[0,k_index,:,:,e_index]= (1/par.iNj)*(1/grids.vG.size)*mPi_joint[:,e_index]     
                    ce_renter_newborns[wf_idx,t_index,k_index,e_index] = np.sum(mDist1_renter[0,k_index, :, :,e_index]* v_nonowner_wf_expanded_pol[t_index,0,k_index, :, :,e_index])-wf_pol_newborns[t_index,k_index,e_index]
            if t_index<grids.vTime.size-1:
                coastal_beq_BAU=vcoastal_beq_BAU[t_index]
                noncoastal_beq_BAU=vnoncoastal_beq_BAU[t_index]
                savings_beq_BAU=vsavings_beq_BAU[t_index] 
                        
    par.wf_wedge[0] = 0.0
    
    
    tax_equiv_C = np.zeros((k_dim, grids.vE.size))
    tax_equiv_NC = np.zeros((k_dim, grids.vE.size))
    tax_equiv_renter = np.zeros((k_dim, grids.vE.size))
    tax_equiv_newborns = np.zeros((grids.vTime.size, k_dim, grids.vE.size))
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):            
            
            tax_equiv_C[k_index, e_index] = find_zero_linear(wf_loss, ce_C[:,k_index, e_index]) # find wf_loss that makes ce_C[:,k_index, g_index, e_index] equal to zero
            tax_equiv_NC[k_index, e_index] = find_zero_linear(wf_loss, ce_NC[:,k_index, e_index])  # find wf_loss that makes ce_NC[:,k_index, g_index, e_index] equal to zero
            tax_equiv_renter[k_index, e_index] = find_zero_linear(wf_loss, ce_renter[:,k_index, e_index])  # find wf_loss that makes ce_renter[:,k_index, g_index, e_index] equal to zero

            for t_index in range(grids.vTime.size):
                tax_equiv_newborns[t_index, k_index, e_index] = find_zero_linear(wf_loss, ce_renter_newborns[:,t_index, k_index, e_index])
                    

    return tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns

@njit
def find_zero_linear(xgrid, ygrid):
    n = xgrid.size

    # exact hit
    for i in range(n):
        if ygrid[i] == 0.0:
            return xgrid[i]

    # sign change between adjacent points
    for i in range(n - 1):
        y0 = ygrid[i]
        y1 = ygrid[i + 1]

        if (y0 < 0.0 and y1 > 0.0) or (y0 > 0.0 and y1 < 0.0):
            x0 = xgrid[i]
            x1 = xgrid[i + 1]
            return x0 - y0 * (x1 - x0) / (y1 - y0)

    # no crossing found
    return np.nan

@njit 
def grid_adjust_rentshape(par,grids,v_nonowner_in):
    T,J,K,G,X,E= v_nonowner_in.shape 
    v_rentshape_expanded=np.zeros((T,J,K,G, grids.vX_sim.size, E))
    for x_index_sim in range(grids.vX_sim.size):
        x=grids.vX_sim[x_index_sim]/(1+par.r) #Need to correct for inconsistency between VFI and simulation in how we record savings (excluding interest rate in VFI, including interest rate in simulation)
        x_index_l=misc.binary_search(0, grids.vX.size, grids.vX,x) 
        x_weight_left=compute_p_left(grids.vX, x, x_index_l)
        v_rentshape_expanded[:,:,:,:,x_index_sim,:]+=x_weight_left*v_nonowner_in[:,:,:,:,x_index_l,:]
        v_rentshape_expanded[:,:,:,:,x_index_sim,:]+=(1-x_weight_left)*v_nonowner_in[:,:,:,:,x_index_l+1,:]  
    return v_rentshape_expanded


@njit 
def grid_adjust(par,grids,v_owner_in):
    T,J,K,G,M,H,L,E= v_owner_in.shape
    v_stay_expanded=np.zeros((T,J,K,G, grids.vM_sim.size, H, grids.vL_sim.size, E))

    for m_index_sim in range(grids.vM_sim.size):
        m=grids.vM_sim[m_index_sim]/(1+par.r) #Need to correct for inconsistency between VFI and simulation in how we record savings (excluding interest rate in VFI, including interest rate in simulation)
        m_index_l=misc.binary_search(0, grids.vM.size, grids.vM,m) 
        m_weight_left=compute_p_left(grids.vM, m, m_index_l)
        for l_index_sim in range(grids.vL_sim.size):
            ltv=grids.vL_sim[l_index_sim]
            l_index_l=misc.binary_search(0, grids.vL.size, grids.vL,ltv) 
            l_weight_left=compute_p_left(grids.vL, ltv, l_index_l)
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*l_weight_left*v_owner_in[:,:,:,:,m_index_l,:,l_index_l,:]
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*l_weight_left*v_owner_in[:,:,:,:,m_index_l+1,:,l_index_l,:]
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*(1-l_weight_left)*v_owner_in[:,:,:,:,m_index_l,:,l_index_l+1,:]
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*(1-l_weight_left)*v_owner_in[:,:,:,:,m_index_l+1,:,l_index_l+1,:] 
    return v_stay_expanded

@njit
def compute_p_left(grid, x, i_left):
    
    x_left = grid[i_left]
    x_right = grid[i_left + 1]
    p_left = (x_right - x) / (x_right - x_left)

    return p_left

@njit
def retransform_coeffs(vCoeff_C, vTime_old, vTime_new):
    """
    Recompute Chebyshev coefficients for a left-truncated time grid.
    
    The polynomial value at each t is preserved exactly.
    Assumes the right endpoint is unchanged: vTime_old[-1] == vTime_new[-1]
    """
    a, b  = vTime_old[0], vTime_old[-1]
    a_new = vTime_new[0]
    # b == vTime_new[-1] (unchanged)

    # s = alpha * s' + beta
    alpha = (b - a_new) / (b - a)
    beta  = (a_new - a) / (b - a)

    c = vCoeff_C  # shorthand

    # Expand T_k(alpha * s' + beta) in terms of powers of s', then
    # collect coefficients for [1, s', s'^2, s'^3, s'^4].
    # Using: T1=x, T2=2x²-1, T3=4x³-3x, T4=8x⁴-8x²+1, x = alpha*s'+beta

    al  = alpha
    be  = beta
    al2 = al**2;  al3 = al**3;  al4 = al**4
    be2 = be**2;  be3 = be**3;  be4 = be**4

    # Contribution of each T_k to power basis [p0, p1, p2, p3, p4]:
    #   c[0]*T0: just c[0]
    #   c[1]*T1(al*s'+be) = c[1]*(al*s' + be)
    #   c[2]*T2(al*s'+be) = c[2]*(2(al*s'+be)² - 1)
    #   c[3]*T3(al*s'+be) = c[3]*(4(al*s'+be)³ - 3(al*s'+be))
    #   c[4]*T4(al*s'+be) = c[4]*(8(al*s'+be)⁴ - 8(al*s'+be)² + 1)

    p = np.zeros(5)  # coefficients of [1, s', s'^2, s'^3, s'^4]

    # T0
    p[0] += c[0]

    # T1
    p[0] += c[1] * be
    p[1] += c[1] * al

    # T2 = 2x² - 1
    p[0] += c[2] * (2*be2 - 1)
    p[1] += c[2] * (4*be*al)
    p[2] += c[2] * (2*al2)

    # T3 = 4x³ - 3x
    p[0] += c[3] * (4*be3 - 3*be)
    p[1] += c[3] * (12*be2*al - 3*al)
    p[2] += c[3] * (12*be*al2)
    p[3] += c[3] * (4*al3)

    # T4 = 8x⁴ - 8x² + 1
    p[0] += c[4] * (8*be4 - 8*be2 + 1)
    p[1] += c[4] * (32*be3*al - 16*be*al)
    p[2] += c[4] * (48*be2*al2 - 8*al2)
    p[3] += c[4] * (32*be*al3)
    p[4] += c[4] * (8*al4)

    # Now p is in the power basis. Convert back to Chebyshev basis.
    # Invert: [T0,T1,T2,T3,T4] -> [1, s', s'^2, s'^3, s'^4]
    # Power basis -> Chebyshev basis (same order, upper triangular inverse):
    new_c = np.zeros(5)
    new_c[4] =  p[4] / 8
    new_c[3] =  p[3] / 4
    new_c[2] = (p[2] - 4*new_c[4]) / 2
    new_c[1] =  p[1] - 3*new_c[3]
    new_c[0] =  p[0] + new_c[2] - new_c[4]

    return new_c