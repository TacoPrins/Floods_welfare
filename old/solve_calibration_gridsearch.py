import numpy as np
import time
import equilibrium as equil
import misc_functions as misc
import tauchen as tauch
import grid_creation as grid_creation
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import moments as find_moments
from scipy.optimize import least_squares
import pandas as pd
import LoM_epsilons as lom
import matplotlib.pyplot as plt



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
    #params0=np.array([0.964, 0.18, 100, 10])
    #lb=np.array([0.9, 0.08, 10, 3.7])
    #ub=np.array([0.98, 0.2, 120, 11.7])
    #residuals = ResidualsWithMemory()
    #sol = least_squares(residuals, params0, bounds=(lb, ub),method="trf", diff_step=0.01, verbose=2,max_nfev=30)
    #print(sol)
    #print(sol.x)
    # params=np.zeros((6))
    # vParam_0=np.array([0.945, 0.95, 0.955, 0.96, 0.965]) # beta
    # vParam_1=np.array([0.16, 0.17, 0.18]) # phi
    # vParam_2=np.array([90, 100, 110]) # B (bequest)
    # vParam_3=np.array([6.7, 7.7, 8.7]) # nu
    # vParam_4=np.array([1.01,1.015, 1.02]) # omega_o
    # vParam_5=np.array([0.0075,0.008, 0.0085]) # psi
    
    # nr_iterations=vParam_0.size*vParam_1.size*vParam_2.size*vParam_3.size*vParam_4.size*vParam_5.size
    # res_collect=np.zeros((nr_iterations,6))
    # params_collect=np.zeros((nr_iterations,6))
    # moments_collect=np.zeros((nr_iterations,23))
    
    # dimension
    params = np.zeros((7))
    
    # random ranges 
    param_ranges = {
        "dBeta":    (0.94, 0.965),   
        "dPhi":     (0.10, 0.15),     
        "dNu":    (10.0, 50.0),    
        "dB_bar":      (0.1, 5.0),       
        "dOmega_O": (0.0, 0.01),      
        "dPsi":     (0.003, 0.006),  
        "dOmega_G": (0.02, 0.06),      
    }
    
    # choose number of random draws
    num_samples = 1 # set n somewhere above, e.g. n = 500
    
    # same containers, just sized by num_samples now
    nr_iterations = num_samples
    res_collect = np.zeros((nr_iterations, 7))
    params_collect = np.zeros((nr_iterations, 7))
    moments_collect = np.zeros((nr_iterations, 27))
    
    # draw all random params at once: shape (num_samples, 6)
    param_draws = np.column_stack([
        np.random.uniform(low, high, num_samples)
        for (low, high) in param_ranges.values()
    ])
    
    residuals = ResidualsWithMemory()
    it_counter = 0
    
    # keep the rest of your workflow the same, replacing nested loops with this loop
    for i in range(nr_iterations):
        params[:] = param_draws[i, :]
    
        res_collect[it_counter,:]=residuals(params)
        params_collect[it_counter,:]=params
        moments_collect[it_counter,:]=residuals.last_moments

        print(params_collect[it_counter,:])
        print(res_collect[it_counter,:])
        it_counter+=1
        print('iteration', it_counter)
                        
    print(params_collect)
    print(res_collect)
    df = pd.DataFrame(res_collect)
    #df.to_excel(excel_writer = "/Users/sevimdinlemez/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD UvA/Projects/Climate risk + house prices/Python/Calibration/residuals.xlsx")
    df = pd.DataFrame(params_collect)
    #df.to_excel(excel_writer = "/Users/sevimdinlemez/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD UvA/Projects/Climate risk + house prices/Python/Calibration/params.xlsx")

    # New: combined params + requested moments
    moment_names = [
    "HO_C_share", "HO_NC_share", "R_C_share", "R_NC_share",
    "HO_C_share_before35", "HO_NC_share_before35",
    "HO_C_share_death", "HO_NC_share_death",
    "total_NW_HO_C", "total_NW_HO_NC", "total_NW_R", "total_NW_HO",
    "total_NW_age_15", "total_NW_age_27", "total_NW_all_ages",
    "median_NW_age_15", "median_NW_age_27", "median_NW_all_ages",
    "thirtythree_percentile_NW_age_27", "sixtyseven_percentile_NW_age_27",
    "thirtythree_percentile_NW_age_30", "sixtyseven_percentile_NW_age_30",
    "tenth_percentile_housing", "median_housing", "ninetieth_percentile_housing", "dP_C", "dP_NC"
    ]

    df_params = pd.DataFrame(
        params_collect,
        columns=["beta", "phi", "nu", "B", "omega_o", "psi", "omega_g"]
    )
    df_moments = pd.DataFrame(moments_collect, columns=moment_names)
    df_all = pd.concat([df_params, df_moments], axis=1)
    df_all.to_excel(
        excel_writer="/Users/sevimdinlemez/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD UvA/Projects/Climate risk + house prices/Python/Calibration/check.xlsx",
        index=False
    )
    
    vHR_ages = residuals.HR_ages
    
    plt.plot(np.arange(30), vHR_ages)
    plt.title('homeownership over the lifecycle')
    plt.ylim(0,1)
    plt.show()
    plt.clf()
    
class ResidualsWithMemory:
    def __init__(self):
        #self.vCoeff_C = np.array([0.61124443 , 0., 0., 0., 0.])
        #self.vCoeff_NC = np.array([0.64658599, 0., 0., 0., 0.])
        self.vCoeff_C = np.array([0.6506083 , 0., 0., 0., 0.])
        self.vCoeff_NC = np.array([0.7282835, 0., 0., 0.])
        self.bequest_guess = np.zeros((3))
        self.last_moments = np.zeros(27)
        self.HR_ages = np.zeros((30))
        
    def __call__(self,params):
        print(params)
        vCoeff_C_initial = self.vCoeff_C
        vCoeff_NC_initial = self.vCoeff_NC
        bequest_guess= self.bequest_guess
        dPi_L = 0.01
    
        t0 = time.time()

        res=np.zeros((params.size))
        
        dBeta= 0.940074219 # params[0]
        dPhi= 0.127929688 # params[1]
        dNu= 44.5312500 # params[2] 
        db_bar=  3.18164063 # params[3]
        dOmega= 0.010156250 # params[4]
        dPsi = 0.00481015625 # params[5]
        omega_g = 0.0223437500 # params[6]
           
        time_increment=2
        vPi_S_median=np.array([0.0194,    0.0198,    0.0202,    0.0206,    0.0210,    0.0214,    0.0218,    0.0222,    0.0226,    0.0230,    0.0234,    0.0239,    0.0243,
                               0.0248,    0.0254,    0.0259,    0.0265,    0.0273,    0.0280,   0.0289,    0.0300,    0.0310,    0.0321,    0.0333,    0.0347,    0.0361,
        0.0376,    0.0392,    0.0410,    0.0427,    0.0444,    0.0461,    0.0478,    0.0495,    0.0513,    0.0530,    0.0547,    0.0565,    0.0583,
        0.0601,    0.0619,    0.0637,    0.0654,    0.0672,    0.0690,    0.0708,    0.0726,    0.0744,    0.0762,    0.0780,    0.0798,    0.0816])
        vPi_S_median=1-(1-vPi_S_median)**time_increment    
        
        
        par_dict = {"time_increment": time_increment,
                  "iNj": 30,
                  "j_ret": 23,
                  "dBeta": dBeta**time_increment, 
                  "dDelta": 1-(1-0.015)**time_increment, 
                  "dDelta_deprec_rental": 1-(1-0.015)**time_increment,
                  "dDelta_default": 0,
                  "r": 1.03**time_increment-1, 
                  #"dDelta_rental": 1.038**time_increment-1, #From KMV - r plus psi=0.008
                  "dPsi": dPsi,
                  "r_m": 1.04**time_increment-1, 
                  'vPi_S_median': vPi_S_median,
                  'dPi_S_initial': vPi_S_median[0],
                  "dKappa_sell": 0.07,
                  "dKappa_buy": 0,
                  "dXi_foreclosure": 0.8,
                  "dNu": dNu,
                  "dZeta": 0.01, 
                  "dZeta_fixed": 1/26, 
                  "lambda_pti":0.25,
                  "max_ltv": 0.95,
                  "damage_states": 3,
                  "dLambda": 0.8,
                  "dGamma": 1/1.25,
                  "dSigma": 2,
                  "b_bar":db_bar,
                  "dPhi":  dPhi,
                  "nonlingrid": 1,
                  "nonlingrid_big": 1,  
                  #"iNb_left_tail": 20,
                  #"iNb_left": 50,
                  #"iNb_right": 10,
                  "iNb":60,
                  "iBmin": 0, 
                  "iBmax": min(20, 20*((1-0.964)/(1-0.97348525))),
                  "dZ":0.8,
                  "h_max": 5.15,
                  "dXi_min":1,
                  "dXi_max": 1 + omega_g,
                  "iXin":4,
                  "alpha_0": 0.4,
                  "dRho": 0.97, 
                  "dSigmaeps": 0.20, 
                  "dSigmaeps_trans": 0.05, 
                  "iNumStates":5, #THIS NUMBER MUST BE UNEVEN
                  "iNumTrans":3, 
                  "iM":1,
                  'vAgeEquiv':np.ones(50),
                  'dNC_frac': 0.5,
                  'dC_frac': 0.5,
                  'dTheta': 1.5/2.5, 
                  'dL': 0.311,
                  'dOmega': dOmega,
                  'sd_income_initial': .3228617,
                  'beta0_nowealth': -.9910767, #=Constant in logit regression nowealth=1 on log income in lowest age bracket
                  'beta1_nowealth': -.3417672, #=Coeff on log income (in terms of sds away from the median) in logit regression nowealth=1 on log income in lowest age bracket
                  'var_poswealth': 1.557037, #=Variance log wealth conditional on wealth>0 in lowest age bracket
                  'beta0_age': 2.0017337, #=Constant of fitted line age --> mean log income 
                  'beta1_age': .78795906  , #=Coeff of age for fitted line age --> mean log income 
                  'beta2_age': -.02717023, #=Coeff of age^2 for fitted line age --> mean log income  
                  'beta3_age': .00042535, #=Coeff of age^3 for fitted line age --> mean log income 
                  'beta4_age':  -2.505727467e-06, #=Coeff of age^4 for fitted line age --> mean log income 
                  'corr_poswealth_income': 0.1984, #=Correlation between log income (in terms of sds away from the median) and log wealth conditional on wealth>0 in lowest age bracket
                  'tau_0': 0.75,
                  'tau_1': 0.151
    
                  }
    
        
        par = misc.construct_jitclass(par_dict)
        mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
        grids, mMarkov=grid_creation.create(par)        
        method='secant'
        dP_C_guess, dP_NC_guess, vCoeff_C_new, vCoeff_NC_new, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, no_beq, iteration=equil.initialise_coefficients_initial(par, grids, method, dPi_L, par.iNj, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, bequest_guess)      
        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_initial(grids, par, dPi_L, par.iNj, mMarkov, vCoeff_C_new[0],vCoeff_NC_new[0])
        mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, vcoastal_beq, vnoncoastal_beq, vsavings_beq, no_beq=sim.stat_dist_finder(False, grids, par, mMarkov, dPi_L, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_new,vCoeff_NC_new, np.zeros((3)))
        dP_C_lom=lom.LoM_C(grids,0, vCoeff_C_new)
        dP_NC_lom=lom.LoM_NC(grids,0, vCoeff_NC_new)
        
        #results_matrix[i,2]=no_beq
        HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_9, total_NW_age_17, total_NW_all_ages, median_NW_age_9, median_NW_age_17, median_NW_all_ages, thirtythree_percentile_NW_age_17, sixtyseven_percentile_NW_age_17, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted=find_moments.calc_moments(par, grids, 0, mDist1_c, mDist1_nc,mDist1_renter, grids.vPi_S_median[0], dPi_L, vCoeff_C_new, vCoeff_NC_new)
        #results_matrix[i,3]=total_NW_age_17/total_NW_age_9
        bequest_guess_new=np.zeros((3))
        bequest_guess_new[0]=coastal_beq
        bequest_guess_new[1]=noncoastal_beq
        bequest_guess_new[2]=savings_beq
        t1 = time.time()
        print('Computation time:', t1-t0)
        res[0]=(median_NW_all_ages-1.2)/1.2
        res[1]=(median_housing-0.5)/0.5
        res[2]=(median_NW_age_17/median_NW_age_9-1.51)/1.51
        res[3]=(HO_C_share_death+HO_NC_share_death-0.8)/0.8
        res[4]=(HO_C_share+HO_NC_share-0.66)/0.66
        res[5]=(HO_C_share_before35 + HO_NC_share_before35-0.39)/0.39

        # store all requested moments (excluding last two calc_moments outputs)
        self.last_moments = np.array([
            HO_C_share, HO_NC_share, R_C_share, R_NC_share,
            HO_C_share_before35, HO_NC_share_before35,
            HO_C_share_death, HO_NC_share_death,
            total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO,
            total_NW_age_9, total_NW_age_17, total_NW_all_ages,
            median_NW_age_9, median_NW_age_17, median_NW_all_ages,
            thirtythree_percentile_NW_age_17, sixtyseven_percentile_NW_age_17, 
            thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30,
            tenth_percentile_housing, median_housing, ninetieth_percentile_housing, dP_C_lom,dP_NC_lom
        ])
        
        for j in range(par.iNj):
            self.HR_ages[j] = ( np.sum(mDist1_c[j,]) + np.sum(mDist1_nc[j,]))/( np.sum(mDist1_c[j,]) + np.sum(mDist1_nc[j,]) + np.sum(mDist1_renter[j,]))
        
        
        
        #print(HO_C_share, HO_NC_share, R_C_share, R_NC_share, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_9, total_NW_age_17, total_NW_all_ages, median_NW_age_9, median_NW_age_17, median_NW_all_ages, thirtythree_percentile_NW_age_17, sixtyseven_percentile_NW_age_17, tenth_percentile_housing, median_housing, ninetieth_percentile_housing)
        print("First residual:", res[0])
        print("Second residual:", res[1])
        print("Third residual", res[2])
        print("Fourth residual", res[3])
        print("Fifth residual", res[4])
        print("sixth residual", res[5])
        #self.vCoeff_C = vCoeff_C_new.copy()
        #self.vCoeff_NC = vCoeff_NC_new.copy()
        self.bequest_guess = bequest_guess
        
        return res

if __name__ == "__main__":
    main()
