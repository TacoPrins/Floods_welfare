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
import proper_welfare_debug as welfare_stats
from numba import config
from scipy.stats import norm
import moments as find_moments
import experiments as experiments


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
    #vCoeff_C_initial=np.array([0.72392258, 0.,         0.,         0.,         0.        ])
    #vCoeff_NC_initial=np.array([0.76693964, 0.,         0.,         0.,         0.        ])
    vCoeff_C_initial=np.array([0.69906474, 0.,         0.,         0.,         0.        ])
    vCoeff_NC_initial=np.array([0.78259554, 0.,         0.,         0.,         0.        ])
    vCoeff_C_terminal_RE=np.array([0.58952906 , 0.,0.,0.,0. ])
    vCoeff_NC_terminal_RE=np.array([0.85484033,0.,0.,0.,0.])
    vCoeff_C_terminal_HE=np.array([0.64908636, 0.,0.,0.,0. ])
    vCoeff_NC_terminal_HE=np.array([0.82124315,0.,0.,0.,0.])
    
    #vCoeff_C_initial=np.array([0.56, 0.,         0.,         0.,         0.        ])
    #vCoeff_NC_initial=np.array([0.85, 0.,         0.,         0.,         0.        ])
    #vCoeff_C_initial=np.array([0.6732231829966053, 0.,         0.,         0.,         0.        ])
    #vCoeff_NC_initial=np.array([0.7170390048443882, 0.,         0.,         0.,         0.        ])
    

    method='secant'

    par = misc.construct_jitclass(parfile.par_dict)

    
    
    
    
 
    # create grids
    
    mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
    grids, mMarkov=grid_creation.create(par)
    
    
    #Create initial guess for house prices - coastal price falls one to one with flood risk, and noncoastal rises by less than half 
    #t_cheby=(2*grids.vTime-(grids.vTime[0]+grids.vTime[-1]))/(grids.vTime[-1]-grids.vTime[0])
    #t_1=t_cheby
    #t_2=2*t_cheby**2-1
    #t_3=4*t_cheby**3-3*t_cheby
    #t_4=8*t_cheby**4-8*t_cheby**2+1
    
    #X = np.column_stack((t_1, t_2, t_3, t_4))    
    #X = np.column_stack((np.ones(len(X)), X))
    #y_c=(1-1*(par.vPi_S_median-par.vPi_S_median[0]))*vCoeff_C_initial[0]
    #beta_c = np.linalg.inv(X.T @ X) @ X.T @ y_c
    #y_nc=(1+0.25*(par.vPi_S_median-par.vPi_S_median[0]))*vCoeff_NC_initial[0]
    #beta_nc = np.linalg.inv(X.T @ X) @ X.T @ y_nc

    vCoeff_C=np.array([ 0.66335385, -0.03015386,  0.00541847,  0.00797395,  0.00249396])
    vCoeff_NC=np.array([ 0.81033554,  0.01679082, -0.00574326, -0.00115107,  0.00101112])
    #NOT CONVERGED YET 
    vCoeff_C_RE=np.array([ 0.6355361, -0.05750348,0.00171657, 0.00611094,0.00187107])
    vCoeff_NC_RE=np.array([ 0.82617263, 0.03256824, -0.00530541,-0.00385609,0.00083488])
    
    
    vCoeff_C_experiment=np.array([ 0.62190337, -0.04657477,  0.00822706,  0.00254822,  0.0029312 ])
    vCoeff_NC_experiment=np.array([ 8.36567710e-01,  2.38785227e-02, -3.96488165e-03, -4.07334828e-04, 2.94338367e-03])
    
    method='secant'
    func=False
    initial=True
    # sceptics=False 
    welfare=True
    # run and save SS without welfare
    # v_owner_c_wf_SS, v_owner_nc_wf_SS, v_nonowner_wf_SS, _, _, _=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    # vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf = household_problem.solve(grids, par, par.iNj, mMarkov,vCoeff_C_RE,vCoeff_NC_RE, sceptics, welfare)
    
    tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE =  welfare_stats.find_expenditure_equiv(par,grids,mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C_RE, vCoeff_NC_RE, False)
    tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns =  welfare_stats.find_expenditure_equiv(par,grids,mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC, True)


    plot_tax_equiv(grids, 100*-tax_equiv_C[:,:], 100*-tax_equiv_NC[:,:], 100*-tax_equiv_renter[:,:], 'Expenditure equivalent - Renters')
    plot_tax_equiv_newborns(grids, 100*-tax_equiv_newborns[:,:,:])
    plot_tax_equiv_RE_vs_nonRE(grids,100*-tax_equiv_C,100*-tax_equiv_NC,100*-tax_equiv_renter,100*-tax_equiv_C_RE, 100*-tax_equiv_NC_RE,100*-tax_equiv_renter_RE)

    plot_tax_equiv_newborns_RE_vs_nonRE(grids,100*-tax_equiv_newborns,100*-tax_equiv_newborns_RE)


###########################################################

### start main
if __name__ == "__main__":
    main()

