import numpy as np
import equilibrium as equil


def solve(par, grids, solve_initial_ss_HE, solve_initial_ss_RE, solve_terminal_ss_HE, solve_terminal_ss_RE, solve_terminal_ss_building_rest,solve_terminal_ss_mortgage_premium,find_coeff_path_HE, find_coeff_path_RE, path_until_experiment, find_coeff_buildingrest, find_coeff_mortgageprem):
    
    "coefficients 2 different initial steady states"
    vCoeff_C_initial_HE_guess = np.array([0.63017547, 0., 0., 0., 0.,])
    vCoeff_NC_initial_HE_guess = np.array([0.7047178, 0., 0., 0., 0.,])
    vCoeff_C_initial_RE_guess = np.array([0.63017547, 0., 0., 0., 0.,])
    vCoeff_NC_initial_RE_guess = np.array([0.7047178, 0., 0., 0., 0.,])
    "coefficients for baseline and rational expectation transitions"
    vCoeff_C_HE_guess=np.array([ 0.56776576, -0.06110282,  0.00295317,  0.00840279 , 0.001457 ])
    vCoeff_NC_HE_guess=np.array([ 0.73157062,  0.02379444 ,-0.00153431, -0.00262569,  0.00075946])
    vCoeff_C_RE_guess=np.array([ 0.56776576, -0.06110282,  0.00295317,  0.00840279 , 0.001457 ])
    vCoeff_NC_RE_guess=np.array([ 0.73157062,  0.02379444 ,-0.00153431, -0.00262569,  0.00075946])
    vCoeff_C_BR_guess=np.array([ 0.56776576, -0.06110282,  0.00295317,  0.00840279 , 0.001457 ])
    vCoeff_NC_BR_guess=np.array([ 0.73157062,  0.02379444 ,-0.00153431, -0.00262569,  0.00075946])
    vCoeff_C_MP_guess=np.array([ 0.56776576, -0.06110282,  0.00295317,  0.00840279 , 0.001457 ])
    vCoeff_NC_MP_guess=np.array([ 0.73157062,  0.02379444 ,-0.00153431, -0.00262569,  0.00075946])

    
    ## config solve_initial_ss_RE
    vCoeff_C_initial_RE, vCoeff_NC_initial_RE, mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE, rental_stock_C_initial_RE, rental_stock_NC_initial_RE, coastal_beq_initial_RE, noncoastal_beq_initial_RE, savings_beq_initial_RE  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_initial_RE_guess, vCoeff_NC_initial_RE_guess, solve_initial_ss_RE)
    dP_C_initial_RE=vCoeff_C_initial_RE[0]
    dP_NC_initial_RE=vCoeff_NC_initial_RE[0]
    
    ## config: find_coef_RE
    _, _, vCoeff_C_RE, vCoeff_NC_RE, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_RE_guess, vCoeff_NC_RE_guess,dP_C_initial_RE, dP_NC_initial_RE,mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE, rental_stock_C_initial_RE, rental_stock_NC_initial_RE, coastal_beq_initial_RE, noncoastal_beq_initial_RE, savings_beq_initial_RE,find_coeff_path_RE)
    print('Found RE coeff')
    #We only simulate the RE distributions forwards without policy experiments, so we don't need to keep the initial distributions 
    del mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE
    
    ## config solve_initial_ss
    vCoeff_C_initial_HE, vCoeff_NC_initial_HE,  mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_initial_HE_guess, vCoeff_NC_initial_HE_guess, solve_initial_ss_HE)
    dP_C_initial_HE=vCoeff_C_initial_HE[0]
    dP_NC_initial_HE=vCoeff_NC_initial_HE[0]
    print('Found HE coeff')
    
    ## config: find_coef_baseline  
    _, _, vCoeff_C_HE, vCoeff_NC_HE, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_HE_guess, vCoeff_NC_HE_guess,dP_C_initial_HE, dP_NC_initial_HE,mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE,find_coeff_path_HE)
    print('Found HE coeff transition')
        
    "find distribution in 2026 (experiment year) with generate price path using coefficients from baseline"
    price_history, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, _, _, _, _, _, _=equil.generate_pricepath(grids, par, vCoeff_C_HE, vCoeff_NC_HE, dP_C_initial_HE, dP_NC_initial_HE, mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE, path_until_experiment)
    dP_C_2026=price_history[-2,0]
    dP_NC_2026=price_history[-2,1]
    coastal_beq_2026=vcoastal_beq[-1]
    noncoastal_beq_2026=vnoncoastal_beq[-1]
    savings_beq_2026=vsavings_beq[-1]
    
    #From this point, we are simulating forward from the experiment year, so delete initial distributions
    del mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE 
    print('Found starting dist')
    
    "find coefficients for two experiments using correct initial distributions (2026)"
    _, _, vCoeff_C_BR, vCoeff_NC_BR, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_BR_guess, vCoeff_NC_BR_guess,dP_C_2026, dP_NC_2026,mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, coastal_beq_2026, noncoastal_beq_2026, savings_beq_2026,find_coeff_buildingrest)
    "find coefficients for two experiments using correct initial distributions (2026)"
    _, _, vCoeff_C_MP, vCoeff_NC_MP, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_MP_guess, vCoeff_NC_MP_guess,dP_C_2026, dP_NC_2026,mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, coastal_beq_2026, noncoastal_beq_2026, savings_beq_2026,find_coeff_mortgageprem)
    
    "find coefficients for 4 different terminal steady states"
    vCoeff_C_terminal_RE_guess = np.array([0.58944375, 0., 0., 0., 0.,])
    vCoeff_NC_terminal_RE_guess = np.array([0.85491565, 0., 0., 0., 0.,])
    vCoeff_C_terminal_HE_guess = np.array([0.64583997, 0., 0., 0., 0.,])
    vCoeff_NC_terminal_HE_guess = np.array([0.81916869, 0., 0., 0., 0.,])
    vCoeff_C_terminal_BR_guess = np.array([0.69186954, 0., 0., 0., 0.,])
    vCoeff_NC_terminal_BR_guess = np.array([0.83346934, 0., 0., 0., 0.,])
    vCoeff_C_terminal_MP_guess = np.array([0.64583997, 0., 0., 0., 0.,])
    vCoeff_NC_terminal_MP_guess = np.array([0.81916869, 0., 0., 0., 0.,])
    ## config solve_terminal_ss_baseline
    vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, _, _, _, _, _, _, _, _   = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_HE_guess, vCoeff_NC_terminal_HE_guess, solve_terminal_ss_HE)

    ## config solve_terminal_ss_RE
    vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, _, _, _, _, _, _, _, _ = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_RE_guess, vCoeff_NC_terminal_RE_guess, solve_terminal_ss_RE)

    ## config solve_terminal_ss_building_rest
    vCoeff_C_terminal_BR, vCoeff_NC_terminal_BR, _, _, _, _, _, _, _, _  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_BR_guess, vCoeff_NC_terminal_BR_guess, solve_terminal_ss_building_rest)

    ## config solve_terminal_ss_mortgage_premium
    vCoeff_C_terminal_MP, vCoeff_NC_terminal_MP, _, _, _, _, _, _, _, _ = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_MP_guess, vCoeff_NC_terminal_MP_guess, solve_terminal_ss_mortgage_premium)
    
    return vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C_initial_RE, vCoeff_NC_initial_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, vCoeff_C_terminal_BR, vCoeff_NC_terminal_BR, vCoeff_C_terminal_MP, vCoeff_NC_terminal_MP, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_HE, vCoeff_NC_HE, vCoeff_C_BR, vCoeff_NC_BR, vCoeff_C_MP, vCoeff_NC_MP