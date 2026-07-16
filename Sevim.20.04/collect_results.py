"""
collect_results.py
"""

###########################################################
### Imports
import numpy as np
from numba import njit
import equilibrium as equil
import proper_welfare_debug as welfare_stats
import plot_creation as plot_creat
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import equilibrium as equil
import LoM_epsilons as lom
import proper_welfare_debug as welfare_stats
import pandas as pd

#@njit
def collect_results(par, grids, vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_MortPrem, vCoeff_NC_MortPrem, vCoeff_C_BuildRest, vCoeff_NC_BuildRest, solve_initial_ss_HE, solve_initial_ss_RE, path_until_experiment,transition_path, transition_path_RE, experiment_building_rest, experiment_mortgage_prem,vCoeff_C_terminal_RE,vCoeff_NC_terminal_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, calculate_welfare):
    
    """
    Inputs: coefficienten voor: (1) initial steady state, (2) baseline transition with sceptics, (3) baseline transition without sceptics (RE), and three sets of policy coefficients for (4) full information shock, (5) building restrictions, (6) mortgage premium
    
    outputs: four distributions over individual state variables over time for the above 5 cases: owners C, owners NC, renters C, renters NC. Perhaps collapsed on the G, J dimensions (if we're not interested in this source of heterogeneity)
    """
    """############################################################################
    ### Plot price paths
    ############################################################################"""
    ### without experiments
    plot_creat.plot_pricepaths(
    par,
    grids,
    vCoeff_C_initial_HE,
    vCoeff_NC_initial_HE,
    vCoeff_C,
    vCoeff_NC,
    vCoeff_C_RE,
    vCoeff_NC_RE,
    vCoeff_C_terminal_RE,
    vCoeff_NC_terminal_RE,
    vCoeff_C_terminal_HE,
    vCoeff_NC_terminal_HE,
)
    # Mortgage premium experiment
    plot_creat.plot_price_transition_exp(
        vCoeff_C_initial_HE,
        vCoeff_NC_initial_HE,
        vCoeff_C,
        vCoeff_NC,
        vCoeff_C_MortPrem,
        vCoeff_NC_MortPrem,
        par,
        grids,
        title="House price transition: mortgage premium experiment",
        switch_index=14,
    )
    
    # Building restriction experiment
    plot_creat.plot_price_transition_exp(
        vCoeff_C_initial_HE,
        vCoeff_NC_initial_HE,
        vCoeff_C,
        vCoeff_NC,
        vCoeff_C_BuildRest,
        vCoeff_NC_BuildRest,
        par,
        grids,
        title="House price transition: building restriction experiment",
        switch_index=14,
    )
    
    """############################################################################
    ### BASELINE ECONOMY 
    ############################################################################"""
    

    "(1) initial steady state"
    
    dP_C_initial = lom.LoM(par,grids,0,vCoeff_C_initial_HE)
    dP_NC_initial = lom.LoM(par,grids,0,vCoeff_NC_initial_HE)

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter,_,_,_ = household_problem.solve_ss(grids, par, vCoeff_C_initial_HE[0], vCoeff_NC_initial_HE[0], solve_initial_ss_HE)
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, no_beq=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial_HE,vCoeff_NC_initial_HE, solve_initial_ss_HE)
    
    "(2) baseline transition with sceptics"
    # run generate price path without experiments, with sceptics. save the (collapsed) distributions. Also save the 2026 distributions and welfare value functions
    _, _, _, _, _, _, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE =equil.generate_pricepath(grids, par, vCoeff_C,vCoeff_NC, dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, transition_path)    
    "(3) baseline transition without sceptics (RE)"
    # run generate price path without experiments, without sceptics. save the (collapsed) distributions and welfare value functions
    price_history_RE, _, _, _, _, _, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, _, _, _, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE = equil.generate_pricepath(grids, par, vCoeff_C_RE, vCoeff_NC_RE, dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, transition_path_RE)
    
    "Plots of distributions"
    ### PLOT - Baseline model dynamics (leverage, savings, sorting, etc.) Use: full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE 
    
    del full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE
    
    if calculate_welfare:
        "WELFARE COSTS OF MISBELIEFS"
        tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE =  welfare_stats.find_expenditure_equiv_EK_SLR(par, grids, vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C_RE, vCoeff_NC_RE, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE, solve_initial_ss_HE, transition_path_RE)
        tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns             =  welfare_stats.find_expenditure_equiv_EK_SLR(par, grids, vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C, vCoeff_NC, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS,  vcoastal_beq, vnoncoastal_beq, vsavings_beq, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, solve_initial_ss_HE, transition_path)
    
    
    """############################################################################
    ### Plot 2026 dist
    ############################################################################"""
    price_history, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, _, _, _, _, _, _, _, _, _, _, _, _=equil.generate_pricepath(grids, par, vCoeff_C, vCoeff_NC, dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, path_until_experiment)
    #plot_creat.plot_distribution_2026(price_history, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, vCoeff_C, vCoeff_NC)
    
    """############################################################################
    ### START EXPERIMENTS
    ############################################################################"""
    if calculate_welfare:
        "(4) + (5) welfare effects of policy: building restrictions and mortgage premium"
        print("start with welfare of policy")
        tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP,tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR = welfare_stats.find_expenditure_equiv_EK_policy(par, grids, vCoeff_C, vCoeff_NC,vCoeff_C_MortPrem, vCoeff_NC_MortPrem, vCoeff_C_BuildRest, vCoeff_NC_BuildRest, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, transition_path, experiment_mortgage_prem, experiment_building_rest)
    # tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR  = welfare_stats.find_expenditure_equiv_EK_policy(par, grids, vCoeff_C, vCoeff_NC, vCoeff_C_BuildRest, vCoeff_NC_BuildRest, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, vTotal_bequest, transition_path, experiment_building_rest)
    # tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP  = welfare_stats.find_expenditure_equiv_EK_policy(par, grids, vCoeff_C, vCoeff_NC, vCoeff_C_MortPrem, vCoeff_NC_MortPrem, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, vTotal_bequest, transition_path, experiment_mortgage_prem)
     
    ### PLOT - WELFARE EFFECTS OF POLICIES (4), (5), COMPARED TO BASELINE IN (2)
    
    ### PLOT SORTING AFTER POLICY CHANGES: EXTENSIVE VERSUS INTENSIVE MARGIN PER TYPE, INCOME
    
    return tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE, tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns, tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR , tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP

def tax_equiv_to_long(
    array,
    scenario,
    household,
    grids,
    par,
):
    """
    Convert either a (K, E) or (T, K, E) tax-equivalent array
    to a tidy pandas DataFrame.
    """
    array = np.asarray(array)

    k_values = np.asarray(grids.vK)
    e_values = np.asarray(grids.vE)

    if array.ndim == 2:
        # Array dimensions: K x E
        k_dim, e_dim = array.shape

        if e_dim != e_values.size:
            raise ValueError(
                f"{scenario}/{household}: E dimension is {e_dim}, "
                f"but grids.vE has length {e_values.size}."
            )

        k_index, e_index = np.indices(array.shape)

        df = pd.DataFrame(
            {
                "scenario": scenario,
                "household": household,
                "t_index": pd.NA,
                "time": np.nan,
                "year": np.nan,
                "k_index": k_index.ravel(),
                "e_index": e_index.ravel(),
                "tax_equiv": array.ravel(),
            }
        )

        df["k_value"] = k_values[df["k_index"].to_numpy()]
        df["e_value"] = e_values[df["e_index"].to_numpy()]

    elif array.ndim == 3:
        # Array dimensions: T x K x E
        t_dim, k_dim, e_dim = array.shape

        if t_dim != grids.vTime.size:
            raise ValueError(
                f"{scenario}/{household}: T dimension is {t_dim}, "
                f"but grids.vTime has length {grids.vTime.size}."
            )

        if e_dim != e_values.size:
            raise ValueError(
                f"{scenario}/{household}: E dimension is {e_dim}, "
                f"but grids.vE has length {e_values.size}."
            )

        t_index, k_index, e_index = np.indices(array.shape)

        df = pd.DataFrame(
            {
                "scenario": scenario,
                "household": household,
                "t_index": t_index.ravel(),
                "k_index": k_index.ravel(),
                "e_index": e_index.ravel(),
                "tax_equiv": array.ravel(),
            }
        )

        df["time"] = np.asarray(grids.vTime)[
            df["t_index"].to_numpy()
        ]
        
        df["year"] = (
            par.starting_year
            + df["time"].to_numpy() * par.time_increment
        )

        df["k_value"] = k_values[df["k_index"].to_numpy()]
        df["e_value"] = e_values[df["e_index"].to_numpy()]

    else:
        raise ValueError(
            f"Expected a 2D or 3D array, received shape {array.shape}."
        )

    return df[
        [
            "scenario",
            "household",
            "t_index",
            "time",
            "year",
            "k_index",
            "k_value",
            "e_index",
            "e_value",
            "tax_equiv",
        ]
    ]


