import numpy as np
import time
import equilibrium as equil
import misc_functions as misc
import tauchen as tauch
import grid_creation as grid_creation
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import moments as find_moments

def DoubleGrid(vGrid1, vGrid2):
    """Takes two vectors and creates a two-dimensional array of possible combinations
    """
    iN1 = vGrid1.shape[0]
    iN2 = vGrid2.shape[0]

    vGrid2_rep = np.repeat(vGrid2, iN1)
    vGrid1_seq = np.array(list(vGrid1) * iN2)
    m12 = np.array([vGrid1_seq, vGrid2_rep])

    return m12

def main():
    
    num_per_param = 3
    vb_bar=np.linspace(7.7-1,7.7+1,num_per_param)
    vNu=np.linspace(50-25,50+25,num_per_param)    
    params = DoubleGrid(vb_bar, vNu).T
    
    sample_size=num_per_param**(params.shape[1])

    results_matrix = np.column_stack([params, np.zeros((sample_size,2))])
    
    
    
    vCoeff_C_initial=np.array([0.58061689, 0.,         0.,         0.,         0.        ])
    vCoeff_NC_initial=np.array([0.60408363, 0.,         0.,         0.,         0.        ])
    dPi_L = 0.01

    t0 = time.time()
    bequest_guess=np.zeros((3))
    
    for i in range(0,sample_size):

        b_bar=results_matrix[i,0]
        dNu=results_matrix[i,1]
               
        par_dict = {"time_increment": 3,
                  "dBeta": 0.964, #0.964**3,
                  "dDelta": 0.015, # 1-(1-0.015)**3,
                  "dDelta_rental": .05, #1.05**3-1,
                  "dDelta_deprec_rental": 0.015,# 1-(1-0.015)**3,
                  "dDelta_default": 0,
                  "r": 0.03, #1.03**3-1,
                  "r_m": 0.04, #1.04**3-1,
                  'dPi_S_initial': 0.0201, #1-(1-0.0201)**3,
                  "dKappa_sell": 0.07,
                  "dKappa_buy": 0,
                  "dXi_foreclosure": 0.8,                  
                  "dNu": dNu,    
                  "dZeta": 0.01, 
                  "dZeta_fixed": 1/26, 
                  "lambda_ltv": 0.95,
                  "lambda_pti":0.25,
                  "iNj": 20,
                  "j_ret": 15,
                  "damage_states": 3,
                  "dLambda": 0.8,
                  "dGamma": 1/1.25,
                  "dSigma": 2,
                  "b_bar":b_bar,                  
                  "dPhi": 0.16,
                  "nonlingrid": 1,
                  "nonlingrid_big": 1,  
                  "iNb_left": 50,
                  "iNb_right": 20,
                  "iBmin": 0, 
                  "iBmax": 15,
                  'max_ltv': 0.95,
                  #"dZ":0.8,
                  "h_max": 5.15,
                  "dXi_min":1,
                  "dXi_max": 1.1,
                  "iXin":6,
                  #"alpha_0": 0.4,
                  "dRho": 0.97, 
                  "dSigmaeps": 0.20, 
                  "dSigmaeps_trans": 0.05, 
                  "iNumStates":5, #THIS NUMBER MUST BE UNEVEN
                  #"iNumTrans":3, 
                  "iM":1,
                  'vAgeEquiv':np.ones(50),
                  'dNC_frac': 0.5,
                  'dC_frac': 0.5,
                  'dTheta': 1.5/2.5,
                  'dL': 0.311,
                  'dOmega': 1.015,
                  'sd_income_initial': 0.42,
                  'beta0_nowealth': -.9910767, #=Constant in logit regression nowealth=1 on log income in lowest age bracket
                  'beta1_nowealth': -.3417672, #=Coeff on log income (in terms of sds away from the median) in logit regression nowealth=1 on log income in lowest age bracket
                  'var_poswealth': 1.557037, #=Variance log wealth conditional on wealth>0 in lowest age bracket
                  'beta0_age': 10.018623, #=Constant of fitted line age --> mean log income 
                  'beta1_age': .35016282  , #=Coeff of age for fitted line age --> mean log income 
                  'beta2_age': -.06303166, #=Coeff of age^2 for fitted line age --> mean log income  
                  'beta3_age': .00580136, #=Coeff of age^3 for fitted line age --> mean log income 
                  'beta4_age':  -.00020296, #=Coeff of age^4 for fitted line age --> mean log income 
                  'corr_poswealth_income': 0.1984, #=Correlation between log income (in terms of sds away from the median) and log wealth conditional on wealth>0 in lowest age bracket
                  'tau_0': 0.75,
                  'tau_1': 0.151
                  
                  }
        
        par = misc.construct_jitclass(par_dict)
        mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
        grids, mMarkov=grid_creation.create(par)
        method='secant'
        
        dP_C_guess, dP_NC_guess, vCoeff_C_initial, vCoeff_NC_initial, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, no_beq, iteration=equil.initialise_coefficients_initial(par, grids, method, dPi_L, par.iNj, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, bequest_guess)      
        print(vCoeff_C_initial, vCoeff_NC_initial)   
        results_matrix[i,2]=no_beq
        HO_C_share, HO_NC_share, R_C_share, R_NC_share, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_10, total_NW_age_J, total_NW_all_ages, median_NW_age_10, median_NW_age_J, median_NW_all_ages, thirtythree_percentile_NW_age_J, sixtyseven_percentile_NW_age_J, tenth_percentile_housing, median_housing, ninetieth_percentile_housing=find_moments.calc_moments(par, grids, 0, mDist1_c, mDist1_nc,mDist1_renter, grids.vPi_S_median[0], dPi_L, vCoeff_C_initial, vCoeff_NC_initial)
        results_matrix[i,3]=total_NW_age_J/total_NW_age_10
        bequest_guess[0]=coastal_beq
        bequest_guess[1]=noncoastal_beq
        bequest_guess[2]=savings_beq
        
    t1 = time.time()
    print('Computation time:', t1-t0)
    print(results_matrix)
    print(HO_C_share, HO_NC_share, R_C_share, R_NC_share, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_10, total_NW_age_J, total_NW_all_ages, median_NW_age_10, median_NW_age_J, median_NW_all_ages, thirtythree_percentile_NW_age_J, sixtyseven_percentile_NW_age_J, tenth_percentile_housing, median_housing, ninetieth_percentile_housing)
    return results_matrix

if __name__ == "__main__":
    main()