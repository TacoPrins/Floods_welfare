import numpy as np
import LoM_epsilons as lom
import tauchen as tauch
import par_epsilons as parfile
import misc_functions as misc
import matplotlib.pyplot as plt
import grid_creation as grid_creation

def main():
    # import parameters
    vCoeff_C_initial=np.array([0.72414899, 0.,         0.,         0.,         0.        ])
    vCoeff_NC_initial=np.array([0.76734488, 0.,         0.,         0.,         0.        ])
    vCoeff_C_RE=np.array([ 0.66131548, -0.05658024,  0.00335141,  0.00720003,  0.00200794])
    vCoeff_NC_RE=np.array([ 8.12788151e-01,  3.63663858e-02, -3.01088988e-03, -4.55020052e-03,
     -2.91834972e-04])
    dP_C_end=0.61663945 
    dP_NC_end=0.8442203 
    
    
    vCoeff_C=np.array([ 0.68326758, -0.04020328, -0.00154051, 0.00402462, 0.00153142])
    vCoeff_NC=np.array([ 7.98521250e-01, 2.38076699e-02, -1.65203993e-03, -1.06683524e-03,
      2.97454569e-04])
    par = misc.construct_jitclass(parfile.par_dict)
    mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
    grids, mMarkov=grid_creation.create(par)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(9, 5))
    
    # Compute trajectories
    yC_RE = lom.LoM_C(grids, grids.vTime, vCoeff_C_RE)
    yNC_RE = lom.LoM_NC(grids, grids.vTime, vCoeff_NC_RE)
    
    yC_HE = lom.LoM_C(grids, grids.vTime, vCoeff_C)
    yNC_HE = lom.LoM_NC(grids, grids.vTime, vCoeff_NC)
    
    # Rescale time: t=0 -> 1998, step=2 years
    years = 1998 + 2 * grids.vTime
    
    # Plot
    lineC, = plt.plot(years, yC_RE, label='Coastal price trajectory, RE', linewidth=2)
    lineNC, = plt.plot(years, yNC_RE, label='Inland price trajectory, RE', linewidth=2)
    
    
    plt.plot(years, yC_HE,
         linestyle=':',
         linewidth=2,
         color=lineC.get_color(),
         label='Coastal price trajectory, HE')

    plt.plot(years, yNC_HE,
         linestyle=':',
         linewidth=2,
         color=lineNC.get_color(),
         label='Inland price trajectory, HE')
    
    # Initial points
    x0_year = 1998
    y_coastal = vCoeff_C_initial[0]
    y_inland = vCoeff_NC_initial[0]
    
    plt.scatter([x0_year], [y_coastal], zorder=5)
    plt.scatter([x0_year], [y_inland], zorder=5)

    
    # Annotations above dots
    plt.annotate("Initial coastal price",
                 (x0_year, y_coastal),
                 xytext=(10, 10),
                 textcoords="offset points",
                 ha='center', fontsize=9)
    
    plt.annotate("Initial inland price",
                 (x0_year, y_inland),
                 xytext=(10, 10),
                 textcoords="offset points",
                 ha='center', fontsize=9)
    
    xT_year = years[-1]

    # Terminal prices (given)
    yC_terminal = dP_C_end
    yNC_terminal = dP_NC_end
    
    # Scatter terminal points
    plt.scatter([xT_year], [yC_terminal], 
                color=lineC.get_color(), zorder=5)
    
    plt.scatter([xT_year], [yNC_terminal], 
                color=lineNC.get_color(), zorder=5)
        
    # Annotations
    plt.annotate("Terminal coastal price",
                 (xT_year, yC_terminal),
                 xytext=(-10, 10),
                 textcoords="offset points",
                 ha='right', fontsize=9)
    
    plt.annotate("Terminal inland price",
                 (xT_year, yNC_terminal),
                 xytext=(-10, 10),
                 textcoords="offset points",
                 ha='right', fontsize=9)
    
    # Dotted guide lines
    plt.vlines(x0_year, y_coastal, yC_RE[0], linestyles='dotted', linewidth=1)
    plt.vlines(x0_year, y_inland, yNC_RE[0], linestyles='dotted', linewidth=1)
    
    # Axis ticks: start at 2000, every 4 years for readability
    start_year = 2000
    end_year = int(years[-1])
    xticks = np.arange(start_year, end_year + 1, 20)
    plt.xticks(xticks)
    
    # Labels & title
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.title("House price trajectories")
    
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()  
    
### start main
if __name__ == "__main__":
    main()
