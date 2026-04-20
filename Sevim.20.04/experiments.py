import numpy as np
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import LoM_epsilons as lom
import equilibrium as equil
import grid_creation as grid_creation
from numba import njit


@njit
def starting_distribution(grids, par, func, mMarkov, vCoeff_in_C,vCoeff_in_NC, vCoeff_C_initial, vCoeff_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, method):
    sceptics=True
    
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf = household_problem.solve(grids, par, par.iNj, mMarkov,vCoeff_in_C,vCoeff_in_NC, sceptics)
    dP_C_lag=vCoeff_C_initial[0]
    dP_NC_lag=vCoeff_NC_initial[0]
    
    price_history=np.zeros((int((2026-1998)/par.time_increment)+1,2))
        
    for t_index in range(int((2026-1998)/par.time_increment)+1):  

        if t_index==0:
            guess_c = lom.LoM_C(grids,t_index,vCoeff_in_C)
            guess_nc = lom.LoM_NC(grids,t_index,vCoeff_in_NC)

        else:
            guess_c = lom.LoM_C(grids,t_index, vCoeff_in_C)+(price_history[t_index-1,0]-lom.LoM_C(grids,t_index-1, vCoeff_in_C))
            guess_nc = lom.LoM_NC(grids,t_index, vCoeff_in_NC)+(price_history[t_index-1,1]-lom.LoM_NC(grids,t_index-1, vCoeff_in_NC))
        bound_c_l= 0.1
        bound_nc_l= 0.1 
        
        bound_c_l_bis=guess_c-0.1
        bound_c_r_bis=guess_c+0.1
        bound_nc_l_bis=guess_nc-0.1
        bound_nc_r_bis=guess_nc+0.1            
        
        #start=time.perf_counter()
        
        #print("Coastal bequest in:",coastal_beq0)
        #print("Noncoastal bequest in:",noncoastal_beq0)
        #print("Savings bequest in:", savings_beq0)
        #print("Coastal owner sum in:", np.sum(mDist0_c))
        #print("Noncoastal owner sum in:", np.sum(mDist0_nc))
        #print("Renter sum in:",np.sum(mDist0_renter))
        
        price_history[t_index,0], price_history[t_index,1], it, succes = equil.house_prices_algorithm(sceptics, func, method, grids, par, guess_c, guess_nc, bound_c_l, bound_nc_l, bound_c_l_bis, bound_nc_l_bis, bound_c_r_bis, bound_nc_r_bis, mMarkov, par.iNj,  mDist0_c, mDist0_nc, mDist0_renter, vt_stay_c[t_index,],  vt_stay_nc[t_index,], vt_renter[t_index,], b_stay_c[t_index,],b_stay_nc[t_index,],  b_renter[t_index,], t_index, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, vCoeff_in_C, vCoeff_in_NC, dP_C_lag, dP_NC_lag)
        #end=time.perf_counter()                                        
        #print("MC time",end-start)
       # mc_time+=end-start
        
        #excess_demand_C_flow, excess_demand_NC_flow, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock = sim.excess_demand_continuous(func, price_history[t,7], int(price_history[t,2]), int(price_history[t,3]),int(price_history[t,4]),int(price_history[t,5]),price_history[t,6], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, price_history[t,0], price_history[t,1], vt_stay_nc, vt_stay_c, vt_renter, b_c_stay, b_renter, b_nc_stay,rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C,vCoeff_in_NC)
        print("Time step:",t_index)
        if t_index<int((2026-1998)/par.time_increment):
            #start=time.perf_counter()
            mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, coastal_beq1, noncoastal_beq1, savings_beq1, _ = sim.update_dist_continuous(sceptics, False, 0, func, grids, par, t_index, mMarkov, par.iNj, mDist0_c, mDist0_nc, mDist0_renter, price_history[t_index,0], price_history[t_index,1], vt_stay_c[t_index,], vt_stay_nc[t_index,],  vt_renter[t_index,], b_stay_c[t_index,], b_stay_nc[t_index,], b_renter[t_index,],  coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C,vCoeff_in_NC, dP_C_lag, dP_NC_lag)
            dP_C_lag=price_history[t_index,0]
            dP_NC_lag=price_history[t_index,1]
            #end=time.perf_counter() 
            #print("dist time",end-start)
            #dist_time+=end-start
             
             
            mDist0_c  = (mDist1_c)
            mDist0_nc = (mDist1_nc)
            mDist0_renter = (mDist1_renter)
            rental_stock_C0= (stock_demand_rental_C1)
            rental_stock_NC0= (stock_demand_rental_NC1)
            coastal_beq0 = (coastal_beq1)
            noncoastal_beq0  = (noncoastal_beq1)
            savings_beq0 = (savings_beq1)
    

    
    return price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, coastal_beq1, noncoastal_beq1, savings_beq1

@njit
def full_information_shock(grids, par, method, iNj, mMarkov, vCoeff_C_experiment, vCoeff_NC_experiment, price_history, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq):
     
    mDist_c_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist_nc_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist_renter_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vX_sim.size, grids.vE.size))
    
    
    mDist_c_start[:,0,:,:,:,:,:] = mDist1_c[:,0,:,:,:,:,:]+mDist1_c[:,1,:,:,:,:,:]
    mDist_nc_start[:,0,:,:,:,:,:] = mDist1_nc[:,0,:,:,:,:,:]+mDist1_nc[:,1,:,:,:,:,:]
    mDist_renter_start[:,0,:,:,:] = mDist1_renter[:,0,:,:,:]+mDist1_renter[:,1,:,:,:]
    
    
    dP_C_initial=price_history[0,-2]
    dP_NC_initial=price_history[1,-2]
    sceptics=False
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, iteration, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, method, sceptics, iNj, mMarkov, vCoeff_C_experiment, vCoeff_NC_experiment,dP_C_initial, dP_NC_initial,mDist_c_start, mDist_nc_start, mDist_renter_start, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq)
    
    return dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf


def full_information_experiment(par, func, method,  vCoeff_C, vCoeff_NC, vCoeff_C_experiment, vCoeff_NC_experiment, vCoeff_C_initial, vCoeff_NC_initial):

    grids, mMarkov=grid_creation.create(par)
    initial=True 
    sceptics=True
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_ss(grids, par, par.iNj, mMarkov, vCoeff_C_initial[0],vCoeff_NC_initial[0], initial, sceptics, False)                                              
    mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, _, _, _, _=sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, np.zeros((3)), initial)
    
    price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, coastal_beq, noncoastal_beq, savings_beq=starting_distribution(grids, par, func, mMarkov, vCoeff_C,vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, method)
    experiment=True
    grids_exp, mMarkov=grid_creation.create(par, experiment)
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=full_information_shock(grids_exp, par, method, par.iNj, mMarkov, vCoeff_C_experiment, vCoeff_NC_experiment, price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, coastal_beq, noncoastal_beq, savings_beq)
    return price_history, dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf