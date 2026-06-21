import numpy as np
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import LoM_epsilons as lom
import equilibrium as equil
import grid_creation as grid_creation
from numba import njit


###################################
# Shock functions: find coefficients
###################################

@njit
def full_information_shock(grids, par, vCoeff_C_experiment, vCoeff_NC_experiment, price_history, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq):
     
    mDist_c_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist_nc_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist_renter_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vX_sim.size, grids.vE.size))
    
    
    mDist_c_start[:,0,:,:,:,:,:] = mDist1_c[:,0,:,:,:,:,:]+mDist1_c[:,1,:,:,:,:,:]
    mDist_nc_start[:,0,:,:,:,:,:] = mDist1_nc[:,0,:,:,:,:,:]+mDist1_nc[:,1,:,:,:,:,:]
    mDist_renter_start[:,0,:,:,:] = mDist1_renter[:,0,:,:,:]+mDist1_renter[:,1,:,:,:]
    
    
    dP_C_initial=price_history[0,-2]
    dP_NC_initial=price_history[1,-2]
    sceptics=False
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, iteration, vt_stay_c, vt_stay_nc, vt_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, sceptics, vCoeff_C_experiment, vCoeff_NC_experiment,dP_C_initial, dP_NC_initial,mDist_c_start, mDist_nc_start, mDist_renter_start, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq)
    
    return dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter,  vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf

@njit
def building_restriction_shock(grids, par, vCoeff_C_experiment, vCoeff_NC_experiment, price_history, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq):
    
    dP_C_initial=price_history[0,-2]
    dP_NC_initial=price_history[1,-2]
    sceptics=True
    building_rest=True
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, iteration, vt_stay_c, vt_stay_nc, vt_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, sceptics, vCoeff_C_experiment, vCoeff_NC_experiment,dP_C_initial, dP_NC_initial,mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, building_rest = building_rest)
    
    return dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf

@njit
def mortgage_shock(grids, par, vCoeff_C_experiment, vCoeff_NC_experiment, price_history, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq):
    
    dP_C_initial=price_history[0,-2]
    dP_NC_initial=price_history[1,-2]
    sceptics=True
    mortgage_premium = True
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, iteration, vt_stay_c, vt_stay_nc, vt_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, sceptics, vCoeff_C_experiment, vCoeff_NC_experiment,dP_C_initial, dP_NC_initial,mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, mortgage_premium = mortgage_premium)
    
    return dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter,vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf
   

###################################
# 2026 distribution
################################### 

@njit
def gen_distribution_now(par, grids, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, config):
    
    price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, _, _, _, _, _, _=equil.generate_pricepath(grids, par, vCoeff_C,vCoeff_NC, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, config)
    
    return price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq


###################################
# Run experiments
################################### 

def full_information_experiment(par, vCoeff_C, vCoeff_NC, vCoeff_C_experiment_guess, vCoeff_NC_experiment_guess, vCoeff_C_initial, vCoeff_NC_initial):

    grids=grid_creation.create(par)
    price_history, _, _, _,  mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq=gen_distribution_now(grids, par, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial)
    start_2026=True
    grids_exp=grid_creation.create(par, start_2026)
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=full_information_shock(grids_exp, par, vCoeff_C_experiment_guess, vCoeff_NC_experiment_guess, price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq[-1], vnoncoastal_beq[-1], vsavings_beq[-1])
    return price_history, dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter,  vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf


def building_restriction_experiments(par, vCoeff_C, vCoeff_NC, vCoeff_C_experiment_guess, vCoeff_NC_experiment_guess, vCoeff_C_initial, vCoeff_NC_initial):
    grids=grid_creation.create(par)
    price_history, _, _, _,  mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq=gen_distribution_now(grids, par, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial)
    start_2026=True
    grids_exp=grid_creation.create(par, start_2026)
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment,  vt_stay_c, vt_stay_nc, vt_renter,  vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=building_restriction_shock(grids_exp, par, vCoeff_C_experiment_guess, vCoeff_NC_experiment_guess, price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq[-1], vnoncoastal_beq[-1], vsavings_beq[-1])
    return price_history, dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf
    
def mortgage_experiment(par, vCoeff_C, vCoeff_NC, vCoeff_C_experiment_guess, vCoeff_NC_experiment_guess, vCoeff_C_initial, vCoeff_NC_initial):
    grids=grid_creation.create(par)
    price_history, _, _, _,  mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq=gen_distribution_now(grids, par, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial)
    start_2026=True
    grids_exp=grid_creation.create(par, start_2026)
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment,  vt_stay_c, vt_stay_nc, vt_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=mortgage_shock(grids_exp, par, vCoeff_C_experiment_guess, vCoeff_NC_experiment_guess, price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq[-1], vnoncoastal_beq[-1], vsavings_beq[-1])
    return price_history, dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter,  vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf
   
