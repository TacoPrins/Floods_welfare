"""
solve.py

Purpose:
    Solve the model
"""
###########################################################
### Imports
import numpy as np
import numba as nb
import pandas as pd
import time
import matplotlib.pyplot as plt
from numba import njit
import misc_functions as misc
import grids as grid
import tauchen as tauch
import par_epsilons as parfile
import simulate_initial_joint as initial_joint
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import equilibrium as equil
# import equilibrium_debug as equilibrium_debug
import LoM_epsilons as lom
import quantecon as qe
import utility_epsilons as ut
import interp as interp
import buyer_problem_simulation as buy_sim
import continuation_value_nolearning as continuation_value_epsilons
import stayer_problem as stayer_problem
import stayer_problem_renter as stayer_problem_renter
import buyer_problem_epsilons as buyer_problem_epsilons
import pandas as pd
import grid_creation as grid_creation
#import error_statistics as err
import moments as mom
from numba import config
from scipy.stats import norm
import plot_creation as plot_creat
import moments as find_moments
import experiment_config as experiment_config
import solve_model as solve_model
import collect_results as res

def get_g_colors(grids):
    cmap = plt.get_cmap("tab10")
    return {g: cmap(g % 10) for g in range(grids.vG.size)}

def plot_tax_equiv_newborns(grids, tax_equiv_newborns,
                            k_labels=('Realists', 'Sceptics')):

    selected_g = [0, 3, 6]
    colors = get_g_colors(grids)

    t = np.asarray(grids.vTime)
    year = 1998 + 2 * t

    vals = tax_equiv_newborns[:, :, selected_g]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        ymin, ymax = -0.05, 0.05
    else:
        ymin = vals.min()
        ymax = vals.max()
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
        ymin -= margin
        ymax += margin
    

    for k in range(tax_equiv_newborns.shape[1]):

        fig, ax = plt.subplots(figsize=(6, 5))

        for g in selected_g:
            ax.plot(year,
                    tax_equiv_newborns[:, k, g],
                    color=colors[g],
                    label=rf"$\omega_g = {grids.vG[g]:.2f}$")

        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Year", fontsize=15)
        ax.set_ylabel("Expenditure equivalent (%)", fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.3)

        if k != 0:
            ax.legend(title=r"Amenity $\omega_g$",
                      fontsize=13,
                      title_fontsize=13,
                      loc="upper right")

        fig.tight_layout()
        plt.show()


def plot_tax_equiv(grids, tax_equiv_C, tax_equiv_NC, tax_equiv_renters,
                   title=None, k_labels=('Realists', 'Sceptics')):

    x = np.asarray(grids.vG)

    data_list = [
        (tax_equiv_C, "Flood-exposed, owners"),
        (tax_equiv_NC, "Inland, owners"),
        (tax_equiv_renters, "Renters")
    ]

    vals = np.concatenate([
        tax_equiv_C[np.isfinite(tax_equiv_C)],
        tax_equiv_NC[np.isfinite(tax_equiv_NC)],
        tax_equiv_renters[np.isfinite(tax_equiv_renters)]
    ])

    if vals.size == 0:
        ymin, ymax = -0.05, 0.05
    else:
        ymin = vals.min()
        ymax = vals.max()
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
        ymin -= margin
        ymax += margin

    for data, subtitle in data_list:

        fig, ax = plt.subplots(figsize=(6, 5))
        k_dim = data.shape[0]

        for k in range(k_dim):
            label = k_labels[k] if k < len(k_labels) else f'k = {k}'
            ax.plot(x, data[k, :], label=label)

        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$\omega_g$', fontsize=15)
        ax.set_ylabel('Expenditure equivalent (%)', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.3)

        if subtitle == "Renters":
            ax.legend(title='Belief type (k)',
                      fontsize=13,
                      title_fontsize=13,
                      loc='upper right')

        fig.tight_layout()
        plt.show()
        
def plot_tax_equiv_newborns_RE_vs_nonRE(
        grids,
        tax_equiv_newborns_nonRE,
        tax_equiv_newborns_RE,
        k_labels=('Realists', 'Sceptics')):

    selected_g = [0, 3, 6]
    colors = get_g_colors(grids)

    tax_equiv_newborns_RE = np.asarray(tax_equiv_newborns_RE).squeeze()

    t = np.asarray(grids.vTime)
    year = 1998 + 2 * t

    vals_nonRE = tax_equiv_newborns_nonRE[:, :, selected_g]
    vals_RE = tax_equiv_newborns_RE[:, selected_g]
    vals = np.concatenate([vals_nonRE[np.isfinite(vals_nonRE)],
                           vals_RE[np.isfinite(vals_RE)]])

    if vals.size == 0:
        ymin, ymax = -0.05, 0.05
    else:
        ymin = vals.min()
        ymax = vals.max()
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
        ymin -= margin
        ymax += margin
    

    for k in range(tax_equiv_newborns_nonRE.shape[1]):

        fig, ax = plt.subplots(figsize=(6, 5))

        for g in selected_g:
            color = colors[g]

            ax.plot(year,
                    tax_equiv_newborns_nonRE[:, k, g],
                    linestyle='-',
                    linewidth=2,
                    color=color,
                    label=rf"$\omega_g = {grids.vG[g]:.2f}$, non-RE")

            ax.plot(year,
                    tax_equiv_newborns_RE[:, g],
                    linestyle='--',
                    linewidth=2,
                    color=color,
                    label=rf"$\omega_g = {grids.vG[g]:.2f}$, RE")

        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Year", fontsize=15)
        ax.set_ylabel("Expenditure equivalent (%)", fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.3)

        if k != 0:
            ax.legend(fontsize=11,
                      loc="upper right")

        fig.tight_layout()
        plt.show()
        
def plot_tax_equiv_RE_vs_nonRE(
        grids,
        tax_equiv_C_nonRE,
        tax_equiv_NC_nonRE,
        tax_equiv_renters_nonRE,
        tax_equiv_C_RE,
        tax_equiv_NC_RE,
        tax_equiv_renters_RE,
        k_labels=('Realists', 'Sceptics')):

    x = np.asarray(grids.vG)

    tax_equiv_C_RE = np.asarray(tax_equiv_C_RE).squeeze()
    tax_equiv_NC_RE = np.asarray(tax_equiv_NC_RE).squeeze()
    tax_equiv_renters_RE = np.asarray(tax_equiv_renters_RE).squeeze()

    data_list = [
        (tax_equiv_C_nonRE, tax_equiv_C_RE, "Flood-exposed, owners"),
        (tax_equiv_NC_nonRE, tax_equiv_NC_RE, "Inland, owners"),
        (tax_equiv_renters_nonRE, tax_equiv_renters_RE, "Renters")
    ]

    vals = np.concatenate([
        tax_equiv_C_nonRE[np.isfinite(tax_equiv_C_nonRE)],
        tax_equiv_NC_nonRE[np.isfinite(tax_equiv_NC_nonRE)],
        tax_equiv_renters_nonRE[np.isfinite(tax_equiv_renters_nonRE)],
        tax_equiv_C_RE[np.isfinite(tax_equiv_C_RE)],
        tax_equiv_NC_RE[np.isfinite(tax_equiv_NC_RE)],
        tax_equiv_renters_RE[np.isfinite(tax_equiv_renters_RE)]
    ])

    if vals.size == 0:
        ymin, ymax = -0.05, 0.05
    else:
        ymin = vals.min()
        ymax = vals.max()
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
        ymin -= margin
        ymax += margin

    for data_nonRE, data_RE, subtitle in data_list:

        fig, ax = plt.subplots(figsize=(6, 5))

        for k in range(data_nonRE.shape[0]):
            label = k_labels[k] if k < len(k_labels) else f'k = {k}'
            ax.plot(x, data_nonRE[k, :],
                    linestyle='-',
                    linewidth=2,
                    label=f"{label}, non-RE")

        ax.plot(x, data_RE,
                linestyle='--',
                linewidth=2,
                label='RE')

        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$\omega_g$', fontsize=15)
        ax.set_ylabel('Expenditure equivalent (%)', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.3)

        if subtitle == "Renters":
            ax.legend(fontsize=11, loc='upper right')

        fig.tight_layout()
        plt.show()
        
###########################################################
### main
def main():
    # import parameters

    par = misc.construct_jitclass(parfile.par_dict)
    grids=grid_creation.create(par)
    """Converged coefficients"""
    # initial coefficients
    vCoeff_C_initial_HE  = np.array([0.630064, 0.000000, 0.000000, 0.000000, 0.000000])
    vCoeff_NC_initial_HE = np.array([0.704592, 0.000000, 0.000000, 0.000000, 0.000000])
    
    vCoeff_C_initial_RE  = np.array([0.630064, 0.000000, 0.000000, 0.000000, 0.000000])
    vCoeff_NC_initial_RE = np.array([0.704592, 0.000000, 0.000000, 0.000000, 0.000000])
    
    
    # terminal coefficients
    vCoeff_C_terminal_HE  = np.array([0.521083, 0.000000, 0.000000, 0.000000, 0.000000])
    vCoeff_NC_terminal_HE = np.array([0.756270, 0.000000, 0.000000, 0.000000, 0.000000])
    
    vCoeff_C_terminal_RE  = np.array([0.521041, 0.000000, 0.000000, 0.000000, 0.000000])
    vCoeff_NC_terminal_RE = np.array([0.756200, 0.000000, 0.000000, 0.000000, 0.000000])
    
    vCoeff_C_terminal_BR  = np.array([0.611530, 0.000000, 0.000000, 0.000000, 0.000000])
    vCoeff_NC_terminal_BR = np.array([0.738228, 0.000000, 0.000000, 0.000000, 0.000000])
    
    vCoeff_C_terminal_MP  = np.array([0.504351, 0.000000, 0.000000, 0.000000, 0.000000])
    vCoeff_NC_terminal_MP = np.array([0.752983, 0.000000, 0.000000, 0.000000, 0.000000])
    
    
    # baseline / HE transition coefficients
    vCoeff_C  = np.array([0.567643, -0.060880,  0.002957,  0.008420,  0.001462])
    vCoeff_NC = np.array([0.731398,  0.024103, -0.001520, -0.002620,  0.000751])
    
    
    # rational expectations transition coefficients
    vCoeff_C_RE  = np.array([0.564674, -0.057240,  0.003378,  0.006191,  0.002016])
    vCoeff_NC_RE = np.array([0.731454,  0.024241, -0.001665, -0.002878,  0.000382])
    
    
    # mortgage premium experiment coefficients
    vCoeff_C_MortPrem  = np.array([0.536954, -0.047918, -0.006990,  0.015263, -0.002136])
    vCoeff_NC_MortPrem = np.array([0.715914,  0.033233, -0.008939,  0.002877, -0.000985])
    
    
    # building restriction experiment coefficients
    vCoeff_C_BuildRest  = np.array([0.590399, -0.005430, -0.013503,  0.013706, -0.004146])
    vCoeff_NC_BuildRest = np.array([0.722909,  0.029037, -0.011394,  0.002883, -0.001576])
    

    
        
    """Solve for coefficients"""
    solve_initial_ss_HE = misc.construct_jitclass(experiment_config.solve_initial_ss_HE)
    solve_terminal_ss_HE = misc.construct_jitclass(experiment_config.solve_terminal_ss_HE)
    solve_initial_ss_RE = misc.construct_jitclass(experiment_config.solve_initial_ss_RE)
    solve_terminal_ss_RE = misc.construct_jitclass(experiment_config.solve_terminal_ss_RE)
    solve_terminal_ss_building_rest = misc.construct_jitclass(experiment_config.solve_terminal_ss_building_rest)
    solve_terminal_ss_mortgage_premium = misc.construct_jitclass(experiment_config.solve_terminal_ss_mortgage_premium)
    find_coeff_path_HE = misc.construct_jitclass(experiment_config.find_coeff_path_HE)
    find_coeff_path_RE = misc.construct_jitclass(experiment_config.find_coeff_path_RE)
    path_until_experiment = misc.construct_jitclass(experiment_config.path_until_experiment)
    find_coeff_buildingrest = misc.construct_jitclass(experiment_config.find_coeff_buildingrest) 
    find_coeff_mortgageprem = misc.construct_jitclass(experiment_config.find_coeff_mortgageprem) 
    transition_path = misc.construct_jitclass(experiment_config.transition_path)
    transition_path_RE = misc.construct_jitclass(experiment_config.transition_path_RE)
    experiment_building_rest = misc.construct_jitclass(experiment_config.experiment_building_rest)
    experiment_mortgage_prem = misc.construct_jitclass(experiment_config.experiment_mortgage_prem)
   
    # # solve:
    # vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C_initial_RE, vCoeff_NC_initial_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, vCoeff_C_terminal_BR, vCoeff_NC_terminal_BR, vCoeff_C_terminal_MP, vCoeff_NC_terminal_MP, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_HE, vCoeff_NC_HE, vCoeff_C_BR, vCoeff_NC_BR, vCoeff_C_MP, vCoeff_NC_MP=solve_model.solve(par, grids, solve_initial_ss_HE, solve_initial_ss_RE, solve_terminal_ss_HE, solve_terminal_ss_RE, solve_terminal_ss_building_rest,solve_terminal_ss_mortgage_premium,find_coeff_path_HE,find_coeff_path_RE, path_until_experiment, find_coeff_buildingrest, find_coeff_mortgageprem)
    
    # coefficients = {
    # "initial_HE_C": vCoeff_C_initial_HE,
    # "initial_HE_NC": vCoeff_NC_initial_HE,
    # "initial_RE_C": vCoeff_C_initial_RE,
    # "initial_RE_NC": vCoeff_NC_initial_RE,

    # "terminal_HE_C": vCoeff_C_terminal_HE,
    # "terminal_HE_NC": vCoeff_NC_terminal_HE,
    # "terminal_RE_C": vCoeff_C_terminal_RE,
    # "terminal_RE_NC": vCoeff_NC_terminal_RE,
    # "terminal_BR_C": vCoeff_C_terminal_BR,
    # "terminal_BR_NC": vCoeff_NC_terminal_BR,
    # "terminal_MP_C": vCoeff_C_terminal_MP,
    # "terminal_MP_NC": vCoeff_NC_terminal_MP,

    # "transition_RE_C": vCoeff_C_RE,
    # "transition_RE_NC": vCoeff_NC_RE,
    # "transition_HE_C": vCoeff_C_HE,
    # "transition_HE_NC": vCoeff_NC_HE,
    # "transition_BR_C": vCoeff_C_BR,
    # "transition_BR_NC": vCoeff_NC_BR,
    # "transition_MP_C": vCoeff_C_MP,
    # "transition_MP_NC": vCoeff_NC_MP,
    # }
    
    # df = pd.DataFrame.from_dict(
    #     coefficients,
    #     orient="index",
    #     columns=["coeff_0", "coeff_1", "coeff_2", "coeff_3", "coeff_4"],
    # )
    
    # df.index.name = "coefficient_set"
    
    # excel_path = (
    #     "/Users/sevimdinlemez/Library/Mobile Documents/"
    #     "com~apple~CloudDocs/Documents/PhD UvA/Projects/"
    #     "Climate risk + house prices/Python/coefficients.xlsx"
    # )
    
    # df.to_excel(
    #     excel_writer=excel_path,
    #     sheet_name="coefficients",
    #     index=True,
    # )
    
    # print(df)
    
 
    # create grids
    """Steady state moments"""
    # vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter,_,_,_ = household_problem.solve_ss(grids, par, vCoeff_C_initial_HE[0], vCoeff_NC_initial_HE[0], solve_initial_ss_HE)
    # mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, no_beq=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial_HE,vCoeff_NC_initial_HE, solve_initial_ss_HE)
    # dP_C_lom=lom.LoM(par,grids,0, vCoeff_C_initial_HE)
    # dP_NC_lom=lom.LoM(par, grids,0, vCoeff_NC_initial_HE)
    
    # # MODEL MOMENTS
    # HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted=find_moments.calc_moments(par, grids, 0, mDist1_c, mDist1_nc,mDist1_renter,  vCoeff_C_initial_HE, vCoeff_NC_initial_HE)
    # total_saving_model = median_NW_all_ages
    # NW_decay_model = total_NW_age_27/total_NW_age_15
    # bequest_ineq_model = sixtyseven_percentile_NW_age_30/thirtythree_percentile_NW_age_30
    # homeownership_model = HO_C_share+HO_NC_share
    # price_diff_model = (dP_C_lom-dP_NC_lom)/dP_NC_lom
    # homeownership_young_model = HO_C_share_before35 + HO_NC_share_before35
    # med_housing_model = median_housing
    
  
    
    # # DATA MOMENTS
    # total_saving_data = 1.2
    # NW_decay_data = 1.51
    # bequest_ineq_data = 3.24
    # homeownership = 0.66
    # price_diff = -0.114
    # homeownership_young = 0.39
    # med_housing = 0.5
    
    # print('total_saving_model', total_saving_model, 'data:' , total_saving_data)
    # print('NW_decay_data', NW_decay_model, 'data:', NW_decay_data)
    # print('bequest_ineq_data', bequest_ineq_model, 'data:', bequest_ineq_data)
    # print('homeownership', homeownership_model, 'data:', homeownership)
    # print('total_saving_model', price_diff_model, 'data:', price_diff)
    # print('homeownership_young_model', homeownership_young_model, 'data:', homeownership_young)
    # print('med_housing_model', med_housing_model, 'data:', med_housing)
    
    """Collect results along the transition"""
    calculate_welfare = True # (default is true)
    tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE, tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns, tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR , tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP = res.collect_results(par, grids, vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_MortPrem, vCoeff_NC_MortPrem, vCoeff_C_BuildRest, vCoeff_NC_BuildRest, solve_initial_ss_HE, solve_initial_ss_RE, path_until_experiment,transition_path, transition_path_RE, experiment_building_rest, experiment_mortgage_prem,vCoeff_C_terminal_RE,vCoeff_NC_terminal_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, calculate_welfare)
    results = {
    "RE_C": (
        "rational_expectations",
        "owner_C",
        tax_equiv_C_RE,
    ),
    "RE_NC": (
        "rational_expectations",
        "owner_NC",
        tax_equiv_NC_RE,
    ),
    "RE_renter": (
        "rational_expectations",
        "renter",
        tax_equiv_renter_RE,
    ),
    "RE_newborns": (
        "rational_expectations",
        "newborn",
        tax_equiv_newborns_RE,
    ),

    "HE_C": (
        "baseline_HE",
        "owner_C",
        tax_equiv_C,
    ),
    "HE_NC": (
        "baseline_HE",
        "owner_NC",
        tax_equiv_NC,
    ),
    "HE_renter": (
        "baseline_HE",
        "renter",
        tax_equiv_renter,
    ),
    "HE_newborns": (
        "baseline_HE",
        "newborn",
        tax_equiv_newborns,
    ),

    "BR_C": (
        "building_restrictions",
        "owner_C",
        tax_equiv_C_BR,
    ),
    "BR_NC": (
        "building_restrictions",
        "owner_NC",
        tax_equiv_NC_BR,
    ),
    "BR_renter": (
        "building_restrictions",
        "renter",
        tax_equiv_renter_BR,
    ),
    "BR_newborns": (
        "building_restrictions",
        "newborn",
        tax_equiv_newborns_BR,
    ),

    "MP_C": (
        "mortgage_premium",
        "owner_C",
        tax_equiv_C_MP,
    ),
    "MP_NC": (
        "mortgage_premium",
        "owner_NC",
        tax_equiv_NC_MP,
    ),
    "MP_renter": (
        "mortgage_premium",
        "renter",
        tax_equiv_renter_MP,
    ),
    "MP_newborns": (
        "mortgage_premium",
        "newborn",
        tax_equiv_newborns_MP,
    ),
    }
    
    
    # Convert every result matrix into a tidy DataFrame
    result_dfs = {}
    
    for sheet_name, (scenario, household, array) in results.items():
    
        result_dfs[sheet_name] = res.tax_equiv_to_long(
            array=array,
            scenario=scenario,
            household=household,
            grids=grids,
            par=par,
        )
    
    
    # Combine all matrices into one tidy dataset
    df_all = pd.concat(
        result_dfs.values(),
        ignore_index=True,
    )
    
    
    excel_path = (
        "/Users/sevimdinlemez/Library/Mobile Documents/"
        "com~apple~CloudDocs/Documents/PhD UvA/Projects/"
        "Climate risk + house prices/Python/welfare.xlsx"
    )
    
    
    # Save one combined sheet plus a separate sheet for each result matrix
    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
    
        df_all.to_excel(
            excel_writer=writer,
            sheet_name="all_results",
            index=False,
        )
    
        for sheet_name, df in result_dfs.items():
    
            df.to_excel(
                excel_writer=writer,
                sheet_name=sheet_name,
                index=False,
            )
        
###########################################################

### start main
if __name__ == "__main__":
    main()