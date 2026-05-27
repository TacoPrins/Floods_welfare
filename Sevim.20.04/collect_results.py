"""
collect_results.py
"""

###########################################################
### Imports
import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import equilibrium as equil
import proper_welfare_debug as welfare_stats
import plot_creation as plot_creat
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import equilibrium as equil

@njit
def collect_results(par, grids, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_FullInfo, vCoeff_NC_FullInfo, vCoeff_C_MortPrem, vCoeff_NC_MortPrem, vCoeff_C_BuildRest, vCoeff_NC_BuildRest):
    
    """
    Inputs: coefficienten voor: (1) initial steady state, (2) baseline transition with sceptics, (3) baseline transition without sceptics (RE), and three sets of policy coefficients for (4) full information shock, (5) building restrictions, (6) mortgage premium
    
    outputs: four distributions over individual state variables over time for the above 5 cases: owners C, owners NC, renters C, renters NC. Perhaps collapsed on the G, J dimensions (if we're not interested in this source of heterogeneity)
    """
    ############################################################################
    ### Plot price paths
    ############################################################################
    ### without experiments
    
    ### with experiments
    plot_creat.plot_price_transition_exp(
        vCoeff_C_initial,
        vCoeff_NC_initial,
        vCoeff_C,
        vCoeff_NC,
        vCoeff_C_MortPrem,
        vCoeff_NC_MortPrem,
        par,
        grids,
        'Mortgage premium introduction 2026',
        switch_index=14,
    )
    
    plot_creat.plot_price_transition_exp(
        vCoeff_C_initial,
        vCoeff_NC_initial,
        vCoeff_C,
        vCoeff_NC,
        vCoeff_C_BuildRest,
        vCoeff_NC_BuildRest,
        par,
        grids,
        'Building restriction introduction 2026',
        switch_index=14,
    )
    
    
    ############################################################################
    ### BASELINE ECONOMY 
    ############################################################################
    func=False
    method = 'secant'
    "(1) initial steady state"
    initial = True
    sceptics = True
    bequest_guess=np.zeros((3))

    vt_stay_c_ss, vt_stay_nc_ss, vt_renter_ss, b_stay_c_ss, b_stay_nc_ss, b_renter_ss = household_problem.solve_ss(grids, par, par.iNj, mMarkov, vCoeff_C_initial[0],vCoeff_NC_initial[0], initial, sceptics)
    mDist1_c_ss, mDist1_nc_ss, mDist1_renter_ss, rental_stock_C_ss, rental_stock_NC_ss, coastal_beq_ss, noncoastal_beq_ss, savings_beq_ss, _, _, _, no_beq, coastal_mass_J, noncoastal_mass_J, renter_mass_J = sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c_ss[0,], vt_stay_nc_ss[0,], vt_renter_ss[0,], b_stay_c_ss[0,], b_stay_nc_ss[0,], b_renter_ss[0,], vCoeff_C_initial,vCoeff_NC_initial, bequest_guess, initial)

    "(2) baseline transition with sceptics"
    experiment = False
    welfare=True
    plot_stocks=True
    building_rest = False
    mortgage_premium = False
    
    ##PROBLEM - HOW TO GENERATE CONSISTENT 
    
    # run generate price path without experiments, with sceptics. save the (collapsed) distributions. Also save the 2026 distributions and welfare value functions
    price_history_BL, mDist1_c_BL, mDist1_nc_BL, mDist1_renter_BL, stock_demand_rental_C_BL, stock_demand_rental_NC_BL, vcoastal_beq_BL, vnoncoastal_beq_BL, vsavings_beq_BL, vt_stay_c_BL, vt_stay_nc_BL, vt_renter_BL, v_owner_c_wf_BL, v_owner_nc_wf_BL, v_nonowner_wf_BL, full_dist_C_BL, full_dist_NC_BL, full_dist_renter_BL = equil.generate_pricepath(grids, par, func, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist1_c_ss, mDist1_nc_ss, mDist1_renter_ss, rental_stock_C_ss, rental_stock_NC_ss, coastal_beq_ss, noncoastal_beq_ss, savings_beq_ss, coastal_mass_J, noncoastal_mass_J, renter_mass_J, method, sceptics, experiment, welfare, plot_stocks, building_rest, mortgage_premium)
    
    "(3) baseline transition without sceptics (RE)"
    # run generate price path without experiments, without sceptics. save the (collapsed) distributions and welfare value functions
    sceptics = False
    price_history_RE, mDist1_c_RE, mDist1_nc_RE, mDist1_renter_RE, stock_demand_rental_C_RE, stock_demand_rental_NC_RE, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, vt_stay_c_RE, vt_stay_nc_RE, vt_renter_RE, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE = equil.generate_pricepath(grids, par, func, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist1_c_ss, mDist1_nc_ss, mDist1_renter_ss, rental_stock_C_ss, rental_stock_NC_ss, coastal_beq_ss, noncoastal_beq_ss, savings_beq_ss, coastal_mass_J, noncoastal_mass_J, renter_mass_J, method, sceptics, experiment, welfare, plot_stocks, building_rest, mortgage_premium)

    ### PLOT - Baseline model dynamics (leverage, savings, sorting, etc.)
    
    ### PLOT - WELFARE COSTS OF MISBELIEFS
    # run these functions except that now the value functions from (3) and (2) respectively should be imported rather than calculated within the 'find_expenditure_equiv_GK_SLR' functions
    # tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE =  welfare_stats.find_expenditure_equiv_GK_SLR(par,grids,mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C_RE, vCoeff_NC_RE, False)
    # tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns =  welfare_stats.find_expenditure_equiv_GK_SLR(par,grids,mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC, True)
    
    ############################################################################
    ### START EXPERIMENTS
    ############################################################################
    experiment = True
    "(4) full information shock"
    # run generate price path with experiments. save the (collapsed) distributions from 2026 onwards, as well as the value functions
    
    "(5) building restrictions"
    # run generate price path with experiments. save the (collapsed) distributions from 2026 onwards, as well as the value functions
    
    "(6) mortgage premium"
    # run generate price path with experiments. save the (collapsed) distributions from 2026 onwards, as well as the value functions
     
    ### PLOT - WELFARE EFFECTS OF POLICIES (4), (5), (6) COMPARED TO BASELINE IN (2)
    
    ### PLOT SORTING AFTER POLICY CHANGES: EXTENSIVE VERSUS INTENSIVE MARGIN PER TYPE, INCOME, ASSET(?)
    
    