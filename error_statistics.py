"""
simulation.py

Purpose:
    Given the solution for the law of motion, generate error statistics 
"""
import numpy as np
from numba import njit
import interp as interp
import LoM_epsilons as lom
import misc_functions as misc
import buyer_problem_simulation as buy_sim
import numba as nb
import utility_epsilons as ut
import mortgage_choice_simulation as mortgage_sim
import mortgage_choice_simulation_exc as mortgage_sim_exc
import simulation as sim
import household_problem_epsilons as household_problem_epsilons
import equilibrium as equil

@njit
def price_path(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, iT, vCoeff_C, vCoeff_NC):
  
    
   
    func='find'
    learning=False 
    welfare=False
    alpha_guess = 0
    vSimulated_eps, vSimulated_alpha, vEps_k = sim.create_path(par, iT, dPi_S, dPi_L, alpha_guess)    
    vLeps=np.zeros((iT))
    vL2eps=np.zeros((iT))
    
    vLeps[1:] = vSimulated_eps[:-1]
    vL2eps[2:] = vSimulated_eps[:-2]
      
       
    # for guess of coefficients, find value functions                
    vt_stay_c, vt_stay_nc, vt_stay_renter, b_c_stay, b_nc_stay, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C,vCoeff_NC, learning, welfare)

    # given value functions, find no flooding stationary distribution given initial alpha    
    dataset=generate_pricepath_fixedshocks(grids, par, func, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC, alpha_guess, iT,vSimulated_eps, vSimulated_alpha, vEps_k, method, learning, welfare)
              
    return dataset


@njit
def generate_pricepath_fixedshocks(grids, par, func, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_in_C,vCoeff_in_NC, initial_alpha, iT,vSimulated_eps, vSimulated_alpha, vEps_k, method, learning, welfare):
    #mc_time=0 
    #dist_time=0 
    dataset = np.zeros((iT, 14), dtype = np.float64)
    dataset[:,2]  = vSimulated_eps
    dataset[1:,3] = vSimulated_eps[:-1]
    dataset[2:,4] = vSimulated_eps[:-2]
    dataset[3:,5] = vSimulated_eps[:-3]
    dataset[:,6]  = vEps_k
    dataset[:,7]  = vSimulated_alpha
    mDist0_c, mDist0_nc, mDist0_renter, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0 = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_in_C,vCoeff_in_NC,vSimulated_alpha[0], learning)

    for t in range(iT):
        
        #Update guess and bounds        
        if t<3 or dataset[t,2]==1 or dataset[t-1,2]==1 or dataset[t-2,2]==1 or dataset[t-3,2]==1:
            guess_c = lom.LoM_C(dataset[t,7], dataset[t,2], dataset[t,3], dataset[t,4], dataset[t,6],vCoeff_in_C)
            guess_nc =  lom.LoM_NC(dataset[t,7], dataset[t,2], dataset[t,3], dataset[t,4], dataset[t,6],vCoeff_in_NC)
        else:
            guess_c = lom.LoM_C(dataset[t,7], dataset[t,2], dataset[t,3], dataset[t,4], dataset[t,6],vCoeff_in_C)+(dataset[t-1,0]-lom.LoM_C(dataset[t-1,7], dataset[t-1,2], dataset[t-1,3], dataset[t-1,4], dataset[t-1,6],vCoeff_in_C))
            guess_nc = lom.LoM_NC(dataset[t,7], dataset[t,2], dataset[t,3], dataset[t,4], dataset[t,6],vCoeff_in_NC)+(dataset[t-1,1]-lom.LoM_NC(dataset[t-1,7], dataset[t-1,2], dataset[t-1,3], dataset[t-1,4], dataset[t-1,6],vCoeff_in_NC))
        bound_c_l= 0.1
        bound_nc_l= 0.1 
        
        bound_c_l_bis=guess_c-0.25
        bound_c_r_bis=guess_c+0.25
        bound_nc_l_bis=guess_nc-0.25
        bound_nc_r_bis=guess_nc+0.25             
        
        dataset[t,10]=lom.LoM_C(dataset[t,7], dataset[t,2], dataset[t,3], dataset[t,4], dataset[t,5],vCoeff_in_C)
        dataset[t,11]=lom.LoM_NC(dataset[t,7], dataset[t,2], dataset[t,3], dataset[t,4], dataset[t,5],vCoeff_in_NC)

        
        #start=time.perf_counter()
        dataset[t,0], dataset[t,1], it, succes = equil.house_prices_algorithm(func, method, grids, par, guess_c, guess_nc, bound_c_l, bound_nc_l, bound_c_l_bis, bound_nc_l_bis, bound_c_r_bis, bound_nc_r_bis, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C, vCoeff_in_NC,dataset[t,7], int(dataset[t,2]), int(dataset[t,3]),int(dataset[t,4]),int(dataset[t,5]),dataset[t,6], learning)
        dataset[t,8], dataset[t,9], net_demand_C, net_demand_NC, dataset[t,12], dataset[t,13], stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, dataset[t,7], int(dataset[t,2]), int(dataset[t,3]),int(dataset[t,4]),int(dataset[t,5]),dataset[t,6], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, dataset[t,10], dataset[t,11], vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C,vCoeff_in_NC, learning)         
        #end=time.perf_counter() 
        #print("MC time",end-start)
       # mc_time+=end-start
        
        #excess_demand_C_flow, excess_demand_NC_flow, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock = sim.excess_demand_continuous(func, dataset[t,7], int(dataset[t,2]), int(dataset[t,3]),int(dataset[t,4]),int(dataset[t,5]),dataset[t,6], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, dataset[t,0], dataset[t,1], vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C,vCoeff_in_NC)
        print("Time step:",t)
        if t<iT-1:
            #start=time.perf_counter()
            mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental1, coastal_beq1, noncoastal_beq1, savings_beq1 = sim.update_dist_continuous(func,dataset[t,7], int(dataset[t,2]), int(dataset[t,3]),int(dataset[t,4]),int(dataset[t,5]),dataset[t,6], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, dataset[t,0], dataset[t,1], vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C,vCoeff_in_NC, learning)
            #end=time.perf_counter() 
            #print("dist time",end-start)
           # dist_time+=end-start
            
            
            mDist0_c  = (mDist1_c)
            mDist0_nc = (mDist1_nc)
            mDist0_renter = (mDist1_renter)
            rental_stock0= (stock_demand_rental1)
            coastal_beq0 = (coastal_beq1)
            noncoastal_beq0  = (noncoastal_beq1)
            savings_beq0 = (savings_beq1)
            
    return dataset

@njit
def prediction_errors(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC):
    iT=100
    
    dataset=price_path(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, iT, vCoeff_C, vCoeff_NC)
    dP_C_vec=dataset[:,0]
    dP_NC_vec=dataset[:,1]
    dP_C_lom_vec=dataset[:,10]
    dP_NC_lom_vec=dataset[:,11]
    indep_C_vec = np.column_stack((np.ones(iT), dP_C_lom_vec))
    indep_NC_vec = np.column_stack((np.ones(iT), dP_NC_lom_vec))
    
    
    beta_C=misc.ols_numba(indep_C_vec, dP_C_vec)
    beta_NC=misc.ols_numba(indep_NC_vec, dP_NC_vec)
    corr_C= (iT * np.sum(dP_C_vec*dP_C_lom_vec)-(np.sum(dP_C_vec)*np.sum(dP_C_lom_vec)))/(np.sqrt((iT*np.sum(dP_C_vec**2) - np.sum(dP_C_vec)**2)*(iT*np.sum(dP_C_lom_vec**2) - np.sum(dP_C_lom_vec)**2)))
    r_sqrd_C = corr_C**2
    corr_NC= (iT * np.sum(dP_NC_vec*dP_NC_lom_vec)-(np.sum(dP_NC_vec)*np.sum(dP_NC_lom_vec)))/(np.sqrt((iT*np.sum(dP_NC_vec**2) - np.sum(dP_NC_vec)**2)*(iT*np.sum(dP_NC_lom_vec**2) - np.sum(dP_NC_lom_vec)**2)))
    r_sqrd_NC = corr_NC**2
    demand_error_C=(dataset[:,8])/(dataset[:,8]+dataset[:,12])
    demand_error_NC=(dataset[:,9])/(dataset[:,9]+dataset[:,13])
    
    return dataset, beta_C, beta_NC, r_sqrd_C, r_sqrd_NC, demand_error_C, demand_error_NC

