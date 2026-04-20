"""
Calibration.py
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numba import njit
import grids as grid
import tauchen as tauch
import misc as misc
import household_problem as household_problem
import nlopt
import simulate as sim
import equilibrium as equilibrium
import moments as mom
import production_side as prod


def DoubleGrid(vA, vH):
    """Returns a iNa*iNh x 2 matrix containing all possible combinations of saving and housing

    Args:
        vA (vector): Grid of possible savings
        vH (vector): Grid of possible house values

    Returns:
        mAH (matrix, float): Grid
    """
    iNa = vA.shape[0]
    iNh = vH.shape[0]

    vH_rep = np.repeat(vH, iNa)
    vA_seq = np.array(list(vA) * iNh)
    mAH = np.array([vA_seq, vH_rep])

    return mAH
    

def calibration_regions(param_ranges, dTau_c, do, n):

    num_samples = n
    param_matrix = np.column_stack([np.random.uniform(low, high, num_samples) for low, high in param_ranges.values()])
    
    vGamma = np.linspace(0.85,0.95,15)
    ve_bar = np.linspace(6,10,15)
    # vA_g = np.linspace(0.04,0.05,20)
    params = DoubleGrid(vGamma, ve_bar).T
    # params = DoubleGrid(vGamma, vA_g)
    # results_matrix = np.column_stack([param_matrix, np.zeros((n,14))])
    results_matrix = np.column_stack([params, np.zeros((n,14))])
    for i in range(0,n):
    # for i in range(250, 400):
        # dBeta_cali= results_matrix[i,0]
        # eta_cali = results_matrix[i,1]
        # b_bar_cali = results_matrix[i,2]
        dGamma_cali = results_matrix[i,0]
        de_bar_cali = results_matrix[i,1]
        # dA_G_cali = results_matrix[i,5]
        # dA_K_ER_cali = results_matrix[i,6]
        # dA_K_EF_cali = results_matrix[i,7]
        # dA_K_EC_cali = results_matrix[i,8]
        # dA_K_FFC_cali = results_matrix[i,9]
        
        # print('run', i, 'with beta, eta, b_bar, gamma, e_bar, A_G, A_K_R, A_K_F, A_EC, A_FFC', dBeta_cali,  eta_cali,b_bar_cali, dGamma_cali, de_bar_cali, dA_G_cali, dA_K_ER_cali, dA_K_EF_cali, dA_K_EC_cali, dA_K_FFC_cali)
        par_dict = {"J": 50,
                    "J_w": 40,
                    "ret_inc": 0.71,
                    "iNt": 250,
                    "n_z": 7,
                    "n_z_trans": 1,
                    "n_phi": 2,
                    "n_a": 80,
                    "n_x": 80,
                    "n_l": 10,
                    "n_m": 80,
                    "n_FF": 1,
                    "nonlingrid": 1.2,
                    "Amin": 0,
                    "Amax": 35,
                    "Xmax": 40,
                    "Mmin": 4,
                    "Mmax": 40,
                    "dRho": 0.98,
                    "dSigmaeps": 0.10,
                    "dSigmaeps_trans": 0.2,
                    "iM": 3,
                    "dBeta": 9.40791016e-01,#dBeta_cali,
                    "dSigma": 2,
                    "dGamma":   dGamma_cali,
                    "dRho_eff": 1/1.82,
                    "dChi_eff": 1/0.47,
                    "e_bar": de_bar_cali,
                    "dDelta0": 0.01,
                    "dDelta1": 0.02,
                    "dEta": 3.14941406e+01, #eta_cali,
                    "b_bar":  3.27392578e+00,#b_bar_cali,
                    "r_m": 0.03,
                    "lambda_ltv": 0.9,
                    "dTau": 0.06,
                    "varphi0": 0.27,
                    "varphi1": 0.27,
                    "dP_FF": 0.002265,
                    "dZeta": 0.3,
                    "dOmega":0.25,
                    "dAlpha_c": 0.33,
                    "dA":1,
                    "dNu": 0.2,
                    "dPsi": 0.1,
                    "dLambda": 1.8, 
                    "dAlpha_R": 0.5,
                    "dA_KF": 1.74580078e+01, #dA_K_EF_cali,
                    #"dA_KC": 0.9461422601798584,
                    "dA_R":  1.0918504532264803, #dA_K_ER_cali,
                    "dA_F": 1,
                    "dA_EC": 1.57529297e-02, #dA_K_EC_cali,
                    "dA_FFC": 4.75683594e-04,#dA_K_FFC_cali,
                    "dL_bar": 1,
                    "dIota": 1.5,
                    'dKappa': 0.2,
                    "delta_cap": 0.09,
                    'dA_B': 2.2,
                    'p0': 0.39,
                    'var_a0': 0.31,
                    'dA_G': 0.0000001,
                    }
        
        par = misc.construct_jitclass(par_dict)
     
        # create grids
        mMarkov, vZ = tauch.tauchen(par.dRho, par.dSigmaeps, par.n_z, par.iM)
        # mMarkov = np.eye(7)
        # mMarkov_trans, vZ_trans = tauch.tauchen(0, par.dSigmaeps_trans, par.n_z_trans, par.iM)
        mMarkov_trans = np.ones((1,1))
        vZ_trans = np.array([0])
        
        # linear now:
        life_cycle = tauch.lifecycle(40, 50, par.J, par)

        vY_notnormalized = np.exp(life_cycle[:, np.newaxis] + vZ)
        vY_notnormalized[par.J_w:,:] = vY_notnormalized[par.J_w:,:]*par.ret_inc

        #mMarkov_FF, vFF = tauch.tauchen(par.dRho_FF, par.dSigmaeps_FF, 5, par.iM_FF)
        vFF = np.array([par.dP_FF])
        mMarkov_FF = np.ones((1,1))
        grids_dict = {"vA": grid.nonlinspace_jit(par.Amin, par.Amax, par.n_a, 1.2),
                      "vZ": vZ,
                      'vY': tauch.normalize_income(mMarkov, vY_notnormalized, par),
                      'vLifeCycle': life_cycle,
                      'vL': grid.nonlinspace_jit(0, par.lambda_ltv*2.4, par.n_l, 1.2),
                      'mMarkov': mMarkov,
                      'vZ_trans': vZ_trans,
                      'mMarkov_trans': mMarkov_trans[0,:],
                      # "vX": grid.nonlinspace_jit(np.min(tauch.normalize_income(mMarkov, vY_notnormalized, par))- par.dDelta1*(1/par.dA_G), par.Xmax, par.n_x, par.nonlingrid),
                      # "vM": grid.nonlinspace_jit(np.min(tauch.normalize_income(mMarkov, vY_notnormalized, par))- par.dDelta1*(1/par.dA_G), par.Mmax, par.n_m, par.nonlingrid),
                      # "vM_sim": grid.nonlinspace_jit(np.min(tauch.normalize_income(mMarkov, vY_notnormalized, par)) - par.dDelta1*(1/par.dA_G), par.Mmax, par.n_m*2, par.nonlingrid),
                      "vX": grid.nonlinspace_jit(0.1, par.Xmax, par.n_x, 1.1),
                      "vM": grid.nonlinspace_jit(0.1, par.Mmax, par.n_m, 1.1),
                      "vM_sim": grid.nonlinspace_jit(0.1, par.Mmax, par.n_m*2, par.nonlingrid),   
                      "vPhi": np.linspace(0,1,par.n_phi),
                      'vXi': np.array([0,1]),
                      'vFF': vFF,
                      'mMarkov_FF': mMarkov_FF}
        grids = misc.construct_jitclass(grids_dict)
        

        # Estimation
        dTau_c  = 0*grids.vFF[0]
        lump_sump_t = 0
        s_GE = 0
        s_durab = 0
        t0 = time.time()

        redist = 'lump_sump'
        
        r, p_e, w, lump_sump_t, s_durab, s_GE, A_KC = equilibrium.find_ptax_brentq_CRS(par, grids, dTau_c, redist, True)
        t1 = time.time()
        results_matrix[i,2] = p_e
        results_matrix[i,3] = r
        results_matrix[i,4] = w
        results_matrix[i,5] = A_KC
        
        K_C, E_C, FF_C, A_KC, w, conv, iters, p_M, M, sE_in, sF_in = prod.solve_nested_energy_block(par, p_e, dTau_c, r)
        
        v_stay, a_stay, e_stay, c_stay, l_stay, v_adjust, phi_adjust, l_adjust, w_tilde_stay  = household_problem.solve(p_e, r, w, dTau_c, lump_sump_t, s_durab, par, grids)
        bequests_initial_guess = 0.03
        initial_dist = sim.initial_income_wealth_dist(par, grids)
        dist1, demand_e_hh, supply_a, demand_d_g, demand_d_b, demand_ff, demand_c, beq, loan_state, adj_mass = equilibrium.stationary_dist(par, grids, r, p_e, w, dTau_c, lump_sump_t, s_durab, v_stay, v_adjust, a_stay, c_stay, e_stay, l_stay, phi_adjust, l_adjust, initial_dist, bequests_initial_guess)
        
        
        K_E_R, K_E_F, FF_E = prod.calculate_electricty_inputdemands(r, dTau_c, s_GE, p_e, par, E_C + demand_e_hh)
        E_R = par.dA_R * K_E_R
        E_F = par.dA_F *(par.dZeta**(1/par.dOmega) * (par.dA_KF*K_E_F)**((par.dOmega-1)/par.dOmega) + (1-par.dZeta)**(1/par.dOmega) * (FF_E)**((par.dOmega-1)/par.dOmega))**(par.dOmega/(par.dOmega-1))
        
        mean_total, mean_bottom20, mean_top20 = mom.moment_enerexp_over_totalexp(par,grids, e_stay, c_stay, dist1, p_e, dTau_c)
        NW_decay = mom.NW_deathoverret(par, grids, a_stay, dist1)
        bequest_ineq, NW_sorted, dens_sorted = mom.bequest_inequality(par, grids, dist1, a_stay, r)
        # retrieve 8 moments
        results_matrix[i,6] = supply_a
        results_matrix[i,7] = NW_decay
        results_matrix[i,8] = bequest_ineq
        results_matrix[i,9] = mean_total
        results_matrix[i,10] = mean_bottom20 / mean_top20
        _,results_matrix[i,11] = mom.moment_phi_shares(dist1)
        results_matrix[i,12] = E_R / (E_R+E_F)
        results_matrix[i,13] =  p_e/par.dP_FF
        results_matrix[i,14] = E_C/(E_C+FF_C)
        results_matrix[i,15] = FF_C/(FF_C + demand_ff +FF_E)
        print('run', i, 'done ')
        
    return results_matrix

def f(x,grad):
    # dbeta, deta, b_bar, dgamma, e_bar, a_g

    dTau_c = 0
    redist = 'lumpsump'
    # Define bounds for each parameter: dbeta, deta, b_bar, dgamma, s_bar, a_g
    par_dict = {"J": 50,
                "J_w": 40,
                "ret_inc": 0.71,
                "iNt": 250,
                "n_z": 7,
                "n_z_trans": 1,
                "n_phi": 2,
                "n_a": 80,
                "n_x": 80,
                "n_l": 10,
                "n_m": 80,
                "n_FF": 1,
                "nonlingrid": 1.2,
                "Amin": 0,
                "Amax": 35,
                "Xmax": 40,
                "Mmin": 4,
                "Mmax": 40,
                "dRho": 0.98,
                "dSigmaeps": 0.10,
                "dSigmaeps_trans": 0.2,
                "iM": 3,
                "dBeta": x[0],
                "dSigma": 2,
                "dGamma":   x[3],
                "dRho_eff": 1/1.82,
                "dChi_eff": 1/0.47,
                "e_bar": x[4],
                "dDelta0": 0.01,
                "dDelta1": 0.02,
                "dEta": x[1],
                "b_bar":  x[2],
                "r_m": 0.03,
                "lambda_ltv": 0.9,
                "dTau": 0.06,
                "varphi0": 0.27,
                "varphi1": 0.27,
                "dP_FF": 0.002265,
                "dZeta": 0.3,
                "dOmega":0.25,
                "dAlpha_c": 0.33,
                "dA":1,
                "dNu": 0.2,
                "dPsi": 0.1,
                "dLambda": 1.8, 
                "dAlpha_R": 0.5,
                "dA_KF": x[6],
                #"dA_KC": 0.9461422601798584,
                "dA_R": x[5],
                "dA_F": 1,
                "dA_EC": x[7],
                "dA_FFC": x[8],
                "dL_bar": 1,
                "dIota": 1.5,
                'dKappa': 0.2,
                "delta_cap": 0.09,
                'dA_B': 2.2,
                'p0': 0.39,
                'var_a0': 0.31,
                'dA_G': 0.0000001,
                }
    par = misc.construct_jitclass(par_dict)
 
 
    par = misc.construct_jitclass(par_dict)
 
    # create grids
    mMarkov, vZ = tauch.tauchen(par.dRho, par.dSigmaeps, par.n_z, par.iM)
    # mMarkov = np.eye(7)
    # mMarkov_trans, vZ_trans = tauch.tauchen(0, par.dSigmaeps_trans, par.n_z_trans, par.iM)
    mMarkov_trans = np.ones((1,1))
    vZ_trans = np.array([0])
    
    # linear now:
    life_cycle = tauch.lifecycle(40, 50, par.J, par)

    vY_notnormalized = np.exp(life_cycle[:, np.newaxis] + vZ)
    vY_notnormalized[par.J_w:,:] = vY_notnormalized[par.J_w:,:]*par.ret_inc

    #mMarkov_FF, vFF = tauch.tauchen(par.dRho_FF, par.dSigmaeps_FF, 5, par.iM_FF)
    vFF = np.array([par.dP_FF])
    mMarkov_FF = np.ones((1,1))
    grids_dict = {"vA": grid.nonlinspace_jit(par.Amin, par.Amax, par.n_a, 1.2),
                  "vZ": vZ,
                  'vY': tauch.normalize_income(mMarkov, vY_notnormalized, par),
                  'vLifeCycle': life_cycle,
                  'vL': grid.nonlinspace_jit(0, par.lambda_ltv*2.4, par.n_l, 1.2),
                  'mMarkov': mMarkov,
                  'vZ_trans': vZ_trans,
                  'mMarkov_trans': mMarkov_trans[0,:],
                  # "vX": grid.nonlinspace_jit(np.min(tauch.normalize_income(mMarkov, vY_notnormalized, par))- par.dDelta1*(1/par.dA_G), par.Xmax, par.n_x, par.nonlingrid),
                  # "vM": grid.nonlinspace_jit(np.min(tauch.normalize_income(mMarkov, vY_notnormalized, par))- par.dDelta1*(1/par.dA_G), par.Mmax, par.n_m, par.nonlingrid),
                  # "vM_sim": grid.nonlinspace_jit(np.min(tauch.normalize_income(mMarkov, vY_notnormalized, par)) - par.dDelta1*(1/par.dA_G), par.Mmax, par.n_m*2, par.nonlingrid),
                  "vX": grid.nonlinspace_jit(0.1, par.Xmax, par.n_x, 1.1),
                  "vM": grid.nonlinspace_jit(0.1, par.Mmax, par.n_m, 1.1),
                  "vM_sim": grid.nonlinspace_jit(0.1, par.Mmax, par.n_m*2, par.nonlingrid),   
                  "vPhi": np.linspace(0,1,par.n_phi),
                  'vXi': np.array([0,1]),
                  'vFF': vFF,
                  'mMarkov_FF': mMarkov_FF}
    grids = misc.construct_jitclass(grids_dict)
    

    # Estimation
    dTau_c  = 0*grids.vFF[0]
    lump_sump_t = 0
    s_GE = 0
    s_durab = 0

    redist = 'lump_sump'
    
    r, p_e, w, lump_sump_t, s_durab, s_GE, A_KC = equilibrium.find_ptax_brentq_CRS(par, grids, dTau_c, redist)

    K_C, E_C, FF_C, A_KC, w, conv, iters, p_M, M, sE_in, sF_in = prod.solve_nested_energy_block(par, p_e, dTau_c, r)
    
    v_stay, a_stay, e_stay, c_stay, l_stay, v_adjust, phi_adjust, l_adjust, w_tilde_stay  = household_problem.solve(p_e, r, w, dTau_c, lump_sump_t, s_durab, par, grids)
    bequests_initial_guess = 0.03
    initial_dist = sim.initial_income_wealth_dist(par, grids)
    dist1, demand_e_hh, supply_a, demand_d_g, demand_d_b, demand_ff, demand_c, beq, loan_state, adj_mass = equilibrium.stationary_dist(par, grids, r, p_e, w, dTau_c, lump_sump_t, s_durab, v_stay, v_adjust, a_stay, c_stay, e_stay, l_stay, phi_adjust, l_adjust, initial_dist, bequests_initial_guess)
    
    
    K_E_R, K_E_F, FF_E = prod.calculate_electricty_inputdemands(r, dTau_c, s_GE, p_e, par, E_C + demand_e_hh)
    E_R = par.dA_R * K_E_R
    E_F = par.dA_F *(par.dZeta**(1/par.dOmega) * (par.dA_KF*K_E_F)**((par.dOmega-1)/par.dOmega) + (1-par.dZeta)**(1/par.dOmega) * (FF_E)**((par.dOmega-1)/par.dOmega))**(par.dOmega/(par.dOmega-1))
    
    mean_total, mean_bottom20, mean_top20 = mom.moment_enerexp_over_totalexp(par,grids, e_stay, c_stay, dist1, p_e, dTau_c)
    NW_decay = mom.NW_deathoverret(par, grids, a_stay, dist1)
    bequest_ineq, NW_sorted, dens_sorted = mom.bequest_inequality(par, grids, dist1, a_stay, r)
    
    # MODEL MOMENTS
    total_saving_model = supply_a
    NW_decay_model = NW_decay
    bequest_ineq_model = bequest_ineq
    mean_enerexpoverinc_model = mean_total
    energexp_low_over_high_model = mean_bottom20 / mean_top20
    green_elec_model = E_R / (E_R+E_F)
    rel_elec_price_model = p_e/par.dP_FF
    share_0_model,share_1_model = mom.moment_phi_shares(dist1)
    input_share_elec_model = E_C/(E_C+FF_C)
    ff_finalgoodshare_model = FF_C/(FF_C + demand_ff +FF_E)
    
    # DATA MOMENTS
    total_saving_data = 4.2
    NW_decay_data = 0.98
    bequest_ineq_data = 5.58
    mean_enerexpoverinc_data = 0.15
    energexp_low_over_high_data = 1.4
    green_elec_data = 0.18
    rel_elec_price_data = 2.4
    # share_1_data = 0.01
    input_share_elec_data = 0.34
    ff_finalgoodshare_data = 0.48
    
    sq_saving       = ((total_saving_data-total_saving_model)/total_saving_data)**2
    sq_nw           = ((NW_decay_data-NW_decay_model)/NW_decay_data)**2
    sq_ineqnw       = ((bequest_ineq_data-bequest_ineq_model)/bequest_ineq_data)**2
    sq_energy_share = ((mean_enerexpoverinc_data - mean_enerexpoverinc_model)/mean_enerexpoverinc_data)**2
    sq_homothet     = ((energexp_low_over_high_data - energexp_low_over_high_model)/energexp_low_over_high_data)**2
    sq_greenelec    = ((green_elec_data - green_elec_model)/green_elec_data)**2
    sq_relelecprice = ((rel_elec_price_data - rel_elec_price_model)/rel_elec_price_data)**2
    # sq_sh           = ((share_1_data - (share_1_model/(share_0_model+share_1_model)))/share_1_data)**2
    sq_elecshare    = ((input_share_elec_data - input_share_elec_model)/input_share_elec_data)**2
    sq_fossilshare  = ((ff_finalgoodshare_data - ff_finalgoodshare_model)/ff_finalgoodshare_data)**2
    # weights = np.array([1])
    
    squaredsum =  sq_saving + sq_nw + sq_ineqnw + sq_energy_share + sq_homothet +  sq_greenelec + sq_relelecprice +  sq_elecshare + sq_fossilshare # + sq_sh
    
    return squaredsum

def main():
    
    n = 225
    
    do = 'nvfi'
    dTau_c  = 0
    
    param_ranges = {
        # "dBeta": (0.94,0.97),
        # "dEta": (20, 45),
        # "b_bar": (0.1,5),
        "dGamma": (0.85, 0.95),
        "e_bar": (4, 8),
        # "dA_G": (0.2,0.3),
        # "dA_K_ER": (12, 18),
        # "dA_K_EF": (12,18),
        # "dA_EC": (0.01, 0.06),
        # "dA_FFC": (0.0001,0.001)
    }
    
    
    
    results_matrix = calibration_regions(param_ranges, dTau_c, do, n)
    
    df = pd.DataFrame(results_matrix)
    df.to_excel(excel_writer = "/Users/sevimdinlemez/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD UvA/Projects/Energy price volatilty + inequality/Model/Steady state solution EGM/Calibration/contourmap.xlsx")
    
    
    # n = 200
    
    # do = 'nvfi'
    # dTau_c  = 0
    
    # param_ranges = {
    #     "dBeta": (0.94,0.97),
    #     "dEta": (20, 45),
    #     "b_bar": (0.1,5),
    #     "dGamma": (0.8, 0.9),
    #     "e_bar": (4, 8),
    #     "dA_G": (0.2,0.3),
    #     "dA_K_ER": (12, 18),
    #     "dA_K_EF": (12,18),
    #     "dA_EC": (0.01, 0.06),
    #     "dA_FFC": (0.0001,0.001)
    # }
    
    
    
    # results_matrix = calibration_regions(param_ranges, dTau_c, do, n)
    
    # df = pd.DataFrame(results_matrix)
    # df.to_excel(excel_writer = "/Users/sevimdinlemez/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD UvA/Projects/Energy price volatilty + inequality/Model/Steady state solution EGM/Calibration/graph_regions2.xlsx")
    
    # t0 = time.time()
    
    # # Define bounds for each parameter: dbeta, deta, b_bar, dgamma, s_bar, A_r, A_kf, A_ec, A_ffc
    # lb = [0.94,  30,  3,    0.85,  4.5, 12,  15,  0.005,  0.0001]  # Lower bounds
    # ub = [0.97,  40,  4.5,   0.9,  6.5, 16,  18,  0.03,  0.001]   # Upper bounds
    
    # # Define optimization problem
    # opt = nlopt.opt(nlopt.G_MLSL_LDS, 9)
    # opt.set_lower_bounds(lb)
    # opt.set_upper_bounds(ub)
    # opt.set_min_objective(f)
    # opt.set_local_optimizer(nlopt.opt(nlopt.LN_NELDERMEAD, 9))
    # opt.set_xtol_rel(1e-6)  # Local convergence criteria
    # opt.set_xtol_abs(1e-6)
    
    # # Set up the multi-start algorithm
    # opt.set_maxeval(150)  # Maximum number of evaluations
    # opt.set_population(500)  # Number of starting points
    
    # # Optimize
    # # x = [dbeta, deta, b_bar, dgamma, s_bar,A_g, A_r, A_kf, A_ec, A_ffc]
    # x_opt = opt.optimize([0.95, 32, 4, 0.88, 6, 14, 16, 0.02, 0.0006])
    # t1 = time.time()
    # print("Calibration time:", t1-t0, ".")
    # # Print results
    # print("Optimized parameters:", x_opt)
    
###########################################################
### start main
if __name__ == "__main__":
    main()
