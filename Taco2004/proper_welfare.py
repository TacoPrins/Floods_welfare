import numpy as np
import misc_functions as misc
import household_problem_epsilons_nolearning as household_problem  
import simulation as sim
import equilibrium as equil
import simulate_initial_joint as initial_joint_sim
from numba import njit

@njit
def initial_welfare(par, grids, mMarkov, vCoeff_C_initial, vCoeff_NC_initial):
    initial=True
    sceptics=True 
    welfare=False
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    bequest_guess=np.zeros((3))
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, _, _, _, _=sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, bequest_guess, initial)

    welfare=True
    v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, _, _, _=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    v_owner_c_wf_expanded=grid_adjust(par,grids,v_owner_c_wf)
    v_owner_nc_wf_expanded=grid_adjust(par,grids,v_owner_nc_wf)
    v_nonowner_wf_expanded=grid_adjust_rentshape(par,grids,v_nonowner_wf)

    #Use this as a benchmark for newborns every t>0
    social_welfare_initial_mat=np.zeros((grids.vK.size,grids.vG.size))
    #Use this as the t=0 benchmark
    social_welfare_renters_mat=np.zeros((grids.vK.size,grids.vG.size))
    social_welfare_coastal_mat=np.zeros((grids.vK.size,grids.vG.size))
    social_welfare_noncoastal_mat=np.zeros((grids.vK.size,grids.vG.size))
    for k_index in range(grids.vK.size):
        for g_index in range(grids.vG.size):
            social_welfare_initial_mat[k_index,g_index]=np.sum(v_nonowner_wf_expanded[0,0,k_index,g_index,]*mDist1_renter[0,])
            social_welfare_renters_mat[k_index,g_index]=np.sum(v_nonowner_wf_expanded[0,:,k_index,g_index,:,:]*mDist1_renter[:,k_index,g_index,:,:])
            social_welfare_coastal_mat[k_index,g_index]=np.sum(v_owner_c_wf_expanded[0,:,k_index,g_index,:,:,:,:]*mDist1_c[:,k_index,g_index,:,:,:,:]) 
            social_welfare_noncoastal_mat[k_index,g_index]=np.sum(v_owner_nc_wf_expanded[0,:,k_index,g_index,:,:,:,:]*mDist1_nc[:,k_index,g_index,:,:,:,:])
    return social_welfare_initial_mat, social_welfare_renters_mat, social_welfare_coastal_mat, social_welfare_noncoastal_mat

@njit
def solve(par, grids, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C_in, vCoeff_NC_in, sceptics=True):
    if sceptics==True:
        k_dim=2
    else:
        k_dim=1
    method='secant'
    func=False
    initial=True
    welfare=False
    # run and save SS without welfare
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    bequest_guess=np.zeros((3))
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, _, _, _, _=sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, bequest_guess, initial)
    
    experiment=False 
    welfare=True
    # save all value functions with welfare
    v_owner_c_wf_SS, v_owner_nc_wf_SS, v_nonowner_wf_SS, _, _, _=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    price_history, _, _, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf=equil.generate_pricepath(grids, par, func, mMarkov, vCoeff_C_in,vCoeff_NC_in, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, method, sceptics, experiment, welfare)
    
    # interpolate to match with M and L dimension
    v_nonowner_wf_expanded=grid_adjust_rentshape(par,grids,v_nonowner_wf)
    v_owner_c_wf_expanded=grid_adjust(par,grids,v_owner_c_wf)
    v_owner_nc_wf_expanded=grid_adjust(par,grids,v_owner_nc_wf)
    v_nonowner_wf_expanded_SS=grid_adjust_rentshape(par,grids,v_nonowner_wf_SS)
    v_owner_c_wf_expanded_SS=grid_adjust(par,grids,v_owner_c_wf_SS)
    v_owner_nc_wf_expanded_SS=grid_adjust(par,grids,v_owner_nc_wf_SS)
    
    x = v_nonowner_wf_expanded_SS[0,0,:,:,:,:]
    y = v_nonowner_wf_expanded[0,0,:,:,:,:]
    
    print("x min/max", np.nanmin(x), np.nanmax(x))
    print("y min/max", np.nanmin(y), np.nanmax(y))
    print("x <= 0:", np.any(x <= 0))
    print("y <= 0:", np.any(y <= 0))
    print("x nan:", np.any(np.isnan(x)), "inf:", np.any(np.isinf(x)))
    print("y nan:", np.any(np.isnan(y)), "inf:", np.any(np.isinf(y)))
    print("ratio < 0:", np.any((x / y) < 0))
    print("y == 0:", np.any(y == 0))
    print("x == 0:", np.any(x == 0))
    

    # NEWBORNS
    welfare_loss_newborns=np.zeros((grids.vTime.size, k_dim, grids.vG.size))
    welfare_loss_newborns_oldweights=np.zeros((grids.vTime.size, k_dim, grids.vG.size))
    for t_index in range(grids.vTime.size):
        dP_C=price_history[t_index,0]
        dP_NC=price_history[t_index,1]
        coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
        housing_bequest=coastal_beq*(1-coastal_damage_frac-par.dDelta)*dP_C + noncoastal_beq*(1-par.dDelta)*dP_NC
        total_bequest = (housing_bequest+savings_beq*(1+par.r))*par.iNj
        mPi_joint=initial_joint_sim.initial_joint(par, grids, total_bequest)
        for k_index in range(k_dim):            
            for g_index in range(grids.vG.size):
                # weight
                mDist1_renter[0,k_index,g_index,:,:]= (1/par.iNj)*(1/grids.vG.size)*mPi_joint
                # CE for t,k,g
                

                CE = (v_nonowner_wf_expanded_SS[0,0,k_index,g_index,:,:]/v_nonowner_wf_expanded[t_index,0,k_index,g_index,:,:])**(1/((1-par.dPhi)*(1-par.dTheta)))-1
                # weighted product
                welfare_loss_newborns[t_index, k_index, g_index] = np.sum(CE*mDist1_renter[0,k_index,g_index,:,:])/np.sum(mDist1_renter[0,k_index,g_index,:,:])
                welfare_loss_newborns_oldweights[t_index, k_index, g_index]=np.sum(CE*mDist1_renter_SS[0,k_index,g_index,:,:])/np.sum(mDist1_renter_SS[0,k_index,g_index,:,:])
        if t_index<grids.vTime.size-1:
            coastal_beq=vcoastal_beq[t_index]
            noncoastal_beq=vnoncoastal_beq[t_index]
            savings_beq=vsavings_beq[t_index]
    
    # ALIVE GENERATIONS
    welfare_loss_alive_C=np.zeros((k_dim, grids.vG.size))
    welfare_loss_alive_NC=np.zeros((k_dim, grids.vG.size))
    welfare_loss_alive_renters=np.zeros((k_dim, grids.vG.size))
    

    welfare_loss_total=np.zeros((k_dim, grids.vG.size))
    
    for k_index in range(k_dim):            
        for g_index in range(grids.vG.size):
            # weights (J,K,G,M,H,L,Z)
            weight_C = mDist1_c_SS[1:,k_index, g_index, :,:,:,:] 
            weight_NC = mDist1_nc_SS[1:,k_index, g_index, :,:,:,:] 
            weight_renter = mDist1_renter_SS[:,k_index, g_index, :,:] 
            # CE for k,g
            wf_C = (v_owner_c_wf_expanded_SS[0,1:,k_index,g_index,:,:,:,:]/v_owner_c_wf_expanded[0,1:,k_index,g_index,:,:,:,:])**(1/((1-par.dPhi)*(1-par.dTheta)))-1
            wf_NC = (v_owner_nc_wf_expanded_SS[0,1:,k_index,g_index,:,:,:,:]/v_owner_nc_wf_expanded[0,1:,k_index,g_index,:,:,:,:])**(1/((1-par.dPhi)*(1-par.dTheta)))-1
            wf_renter = (v_nonowner_wf_expanded_SS[0,:,k_index,g_index,:,:]/v_nonowner_wf_expanded[0,:,k_index,g_index,:,:])**(1/((1-par.dPhi)*(1-par.dTheta)))-1
            # weighted product
            welfare_loss_alive_C[k_index, g_index] = np.sum(weight_C*wf_C)/np.sum(weight_C)
            welfare_loss_alive_NC[k_index, g_index] =np.sum(weight_NC*wf_NC)/np.sum(weight_NC)
            welfare_loss_alive_renters[k_index, g_index] =np.sum(weight_renter*wf_renter)/np.sum(weight_renter)
            
            # TOTAL LOSS: add newborns and alive, no discounting of future
            welfare_loss_total[k_index, g_index] = welfare_loss_alive_C[k_index, g_index] + welfare_loss_alive_NC[k_index, g_index]+welfare_loss_alive_renters[k_index, g_index]+ np.sum(welfare_loss_newborns[:, k_index, g_index])
    
    return welfare_loss_total, welfare_loss_alive_C, welfare_loss_alive_NC, welfare_loss_alive_renters, welfare_loss_newborns, welfare_loss_newborns_oldweights

@njit
def find_expenditure_equiv(par,grids,mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C_in, vCoeff_NC_in):
    method='secant'
    func=False
    initial=True
    welfare=False
    sceptics = True
    experiment = False
    # run and save SS without welfare: get stationary dist
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    bequest_guess=np.zeros((3))
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, _, _, _, _=sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, bequest_guess, initial)
    
    # get value functions over transition with SLR
    welfare = True

    price_history, _, _, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf=equil.generate_pricepath(grids, par, func, mMarkov, vCoeff_C_in,vCoeff_NC_in, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, method, sceptics, experiment, welfare)
    v_nonowner_wf_expanded_SLR=grid_adjust_rentshape(par,grids,v_nonowner_wf)
    v_owner_c_wf_expanded_SLR=grid_adjust(par,grids,v_owner_c_wf)
    v_owner_nc_wf_expanded_SLR=grid_adjust(par,grids,v_owner_nc_wf)
    if sceptics==False:
        k_dim=1
    else:
        k_dim=grids.vK.size        
    wf_loss = np.linspace(0, 0.15,20)
    
    ce_C  = np.zeros((wf_loss.size,k_dim, grids.vG.size, grids.vE.size))
    ce_NC = np.zeros((wf_loss.size,k_dim, grids.vG.size,  grids.vE.size))
    ce_renter = np.zeros((wf_loss.size,k_dim, grids.vG.size,  grids.vE.size))
    ce_renter_newborns = np.zeros((wf_loss.size,grids.vTime.size, k_dim, grids.vG.size,  grids.vE.size))
    for wf_idx in range(wf_loss.size):
        par.wf_wedge[0] = wf_loss[wf_idx]
        v_owner_c_wf_SS, v_owner_nc_wf_SS, v_nonowner_wf_SS, _, _, _ = household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
        v_nonowner_wf_expanded_SS=grid_adjust_rentshape(par,grids,v_nonowner_wf_SS)
        v_owner_c_wf_expanded_SS=grid_adjust(par,grids,v_owner_c_wf_SS)
        v_owner_nc_wf_expanded_SS=grid_adjust(par,grids,v_owner_nc_wf_SS)
        for k_index in range(k_dim):
            for g_index in range(grids.vG.size):
                for e_index in range(grids.vE.size):
                    wf_SS_c = np.sum(mDist1_c_SS[1:,k_index, g_index, :,:,:,e_index]* v_owner_c_wf_expanded_SS[0,1:,k_index, g_index, :,:,:,e_index])/np.sum(mDist1_c_SS[1:,:, :, :,:,:,:])
                    wf_SS_nc = np.sum(mDist1_nc_SS[1:,k_index, g_index, :,:,:,e_index]* v_owner_nc_wf_expanded_SS[0,1:,k_index, g_index, :,:,:,e_index])/np.sum(mDist1_nc_SS[1:,:, :, :,:,:,:])
                    wf_SS_rent = np.sum(mDist1_renter_SS[:,k_index, g_index, :,e_index]* v_nonowner_wf_expanded_SS[0,:,k_index, g_index, :,e_index])/np.sum(mDist1_renter_SS[:,:, :, :,:])
                    
                    wf_SLR_c = np.sum(mDist1_c_SS[1:,k_index, g_index, :,:,:,e_index]* v_owner_c_wf_expanded_SLR[0,1:,k_index, g_index, :,:,:,e_index])/np.sum(mDist1_c_SS[1:,:, :, :,:,:,:])
                    wf_SLR_nc = np.sum(mDist1_nc_SS[1:,k_index, g_index, :,:,:,e_index]* v_owner_nc_wf_expanded_SLR[0,1:,k_index, g_index, :,:,:,e_index])/np.sum(mDist1_nc_SS[1:,:, :, :,:,:,:])
                    wf_SLR_rent = np.sum(mDist1_renter_SS[:,k_index, g_index, :,e_index]* v_nonowner_wf_expanded_SLR[0,:,k_index, g_index, :,e_index])/np.sum(mDist1_renter_SS[:,:, :, :,:])
                    
                    
                    ce_C[wf_idx,k_index, g_index, e_index] = wf_SS_c - wf_SLR_c
                    ce_NC[wf_idx,k_index, g_index, e_index] = wf_SS_nc - wf_SLR_nc
                    ce_renter[wf_idx,k_index, g_index, e_index] = wf_SS_rent - wf_SLR_rent
                    
                    for t_index in range(grids.vTime.size):
                        newborn_sum = 0.0
                        weight_sum = 0.0
                        dP_C=price_history[t_index,0]
                        dP_NC=price_history[t_index,1]
                        coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
                        housing_bequest=coastal_beq*(1-coastal_damage_frac-par.dDelta)*dP_C + noncoastal_beq*(1-par.dDelta)*dP_NC
                        total_bequest = (housing_bequest+savings_beq*(1+par.r))*par.iNj
                        mPi_joint=initial_joint_sim.initial_joint(par, grids, total_bequest)
                        # weight
                        mDist1_renter[0,k_index,g_index,:,:]= (1/par.iNj)*(1/grids.vG.size)*mPi_joint
                        
                        wf_SLR_newborns = np.sum(mDist1_renter[0,k_index, g_index, :,e_index]* v_nonowner_wf_expanded_SLR[t_index,0,k_index, g_index, :,e_index])/np.sum(mDist1_renter[0,:, :, :,:])
                        wf_SS_newborns = np.sum(mDist1_renter_SS[0,k_index, g_index, :,e_index]* v_nonowner_wf_expanded_SS[0,0,k_index, g_index, :,e_index])/np.sum(mDist1_renter_SS[0,:, :, :,:])
                                            
                        ce_renter_newborns[wf_idx,t_index, k_index, g_index, e_index] = wf_SS_newborns - wf_SLR_newborns 
                        if t_index<grids.vTime.size-1:
                            coastal_beq=vcoastal_beq[t_index]
                            noncoastal_beq=vnoncoastal_beq[t_index]
                            savings_beq=vsavings_beq[t_index]
                        
    par.wf_wedge[0] = 0.0
    
    
    tax_equiv_C = np.zeros((k_dim, grids.vG.size, grids.vE.size))
    tax_equiv_NC = np.zeros((k_dim, grids.vG.size, grids.vE.size))
    tax_equiv_renter = np.zeros((k_dim, grids.vG.size, grids.vE.size))
    tax_equiv_newborns = np.zeros((grids.vTime.size, k_dim, grids.vG.size, grids.vE.size))
    
    for k_index in range(k_dim):
        for g_index in range(grids.vG.size):
            for e_index in range(grids.vE.size):
                tax_equiv_C[k_index, g_index, e_index] = find_zero_linear(wf_loss, ce_C[:,k_index, g_index, e_index]) # find wf_loss that makes ce_C[:,k_index, g_index, e_index] equal to zero
                tax_equiv_NC[k_index, g_index, e_index] = find_zero_linear(wf_loss, ce_NC[:,k_index, g_index, e_index])  # find wf_loss that makes ce_NC[:,k_index, g_index, e_index] equal to zero
                tax_equiv_renter[k_index, g_index, e_index] = find_zero_linear(wf_loss, ce_renter[:,k_index, g_index, e_index])  # find wf_loss that makes ce_renter[:,k_index, g_index, e_index] equal to zero
                for t_index in range(grids.vTime.size):
                    tax_equiv_newborns[t_index, k_index, g_index, e_index] = find_zero_linear(wf_loss, ce_renter_newborns[:,t_index, k_index, g_index, e_index])
                    

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