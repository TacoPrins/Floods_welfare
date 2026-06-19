import numpy as np
import equilibrium as equil


def solve(par, grids, solve_initial_ss_HE, solve_initial_ss_RE, solve_terminal_ss_HE, solve_terminal_ss_RE, solve_terminal_ss_building_rest,solve_terminal_ss_mortgage_premium,find_coeff_path_HE, find_coeff_path_RE):
    
    "find coefficients 2 different initial steady states"
    vCoeff_C_initial_HE_guess = np.array([0.69906474, 0., 0., 0., 0.,])
    vCoeff_NC_initial_HE_guess = np.array([0.78259554, 0., 0., 0., 0.,])
    vCoeff_C_initial_RE_guess = np.array([0.69906474, 0., 0., 0., 0.,])
    vCoeff_NC_initial_RE_guess = np.array([0.78259554, 0., 0., 0., 0.,])
    ## config solve_initial_ss
    vCoeff_C_initial_HE, vCoeff_NC_initial_HE,  mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_initial_HE_guess, vCoeff_NC_initial_HE_guess, solve_initial_ss_HE)
    dP_C_initial_HE=vCoeff_C_initial_HE[0]
    dP_NC_initial_HE=vCoeff_NC_initial_HE[0]
    
    ## config solve_initial_ss_RE
    vCoeff_C_initial_RE, vCoeff_NC_initial_RE, mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE, rental_stock_C_initial_RE, rental_stock_NC_initial_RE, coastal_beq_initial_RE, noncoastal_beq_initial_RE, savings_beq_initial_RE  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_initial_RE_guess, vCoeff_NC_initial_RE_guess, solve_initial_ss_RE)
    dP_C_initial_RE=vCoeff_C_initial_RE[0]
    dP_NC_initial_RE=vCoeff_NC_initial_RE[0]
    
    "find coefficients for 4 different terminal steady states"
    vCoeff_C_terminal_RE_guess = np.array([0.58952906, 0., 0., 0., 0.,])
    vCoeff_NC_terminal_RE_guess = np.array([0.85484033, 0., 0., 0., 0.,])
    vCoeff_C_terminal_HE_guess = np.array([0.64908636, 0., 0., 0., 0.,])
    vCoeff_NC_terminal_HE_guess = np.array([0.82124315, 0., 0., 0., 0.,])
    ## config solve_terminal_ss_baseline
    vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, _, _, _, _, _, _, _, _   = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_HE_guess, vCoeff_NC_terminal_HE_guess, solve_terminal_ss_HE)

    ## config solve_terminal_ss_RE
    vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, _, _, _, _, _, _, _, _ = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_RE_guess, vCoeff_NC_terminal_RE_guess, solve_terminal_ss_RE)

    ## config solve_terminal_ss_building_rest
    vCoeff_C_terminal_BR, vCoeff_NC_terminal_BR, _, _, _, _, _, _, _, _  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_RE_guess, vCoeff_NC_terminal_RE_guess, solve_terminal_ss_building_rest)

    ## config solve_terminal_ss_mortgage_premium
    vCoeff_C_terminal_MP, vCoeff_NC_terminal_MP, _, _, _, _, _, _, _, _ = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_RE_guess, vCoeff_NC_terminal_RE_guess, solve_terminal_ss_mortgage_premium)

    "find coefficients for baseline and rational expectation transitions"
    vCoeff_C_HE_guess=np.array([ 0.66335385, -0.03015386,  0.00541847,  0.00797395,  0.00249396])
    vCoeff_NC_HE_guess=np.array([ 0.81033554,  0.01679082, -0.00574326, -0.00115107,  0.00101112])
    vCoeff_C_RE_guess=np.array([ 0.6355361, -0.05750348,0.00171657, 0.00611094,0.00187107])
    vCoeff_NC_RE_guess=np.array([ 0.82617263, 0.03256824, -0.00530541,-0.00385609,0.00083488])
    ## config: find_coef_baseline  
    _, _, vCoeff_C_HE, vCoeff_NC_HE, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_HE_guess, vCoeff_NC_HE_guess,dP_C_initial_HE, dP_NC_initial_HE,mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE,find_coeff_path_HE)
    
    ## confic: find_coef_RE
    _, _, vCoeff_C_RE, vCoeff_NC_RE, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_RE_guess, vCoeff_NC_RE_guess,dP_C_initial_RE, dP_NC_initial_RE,mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE, rental_stock_C_initial_RE, rental_stock_NC_initial_RE, coastal_beq_initial_RE, noncoastal_beq_initial_RE, savings_beq_initial_RE,find_coeff_path_RE)

    "find distribution in 2026 (experiment year) with generate price path using coefficients from baseline"
    
    ## config: baseline_until_experiment
    
    "find coefficients for two experiments using correct initial distributions (2026)"
    
    ## config: find_coef_REfind_coef_buildingrest
    
    ## confic: find_coef_mortgageprem
    return vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C_initial_RE, vCoeff_NC_initial_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, vCoeff_C_terminal_BR, vCoeff_NC_terminal_BR, vCoeff_C_terminal_MP, vCoeff_NC_terminal_MP