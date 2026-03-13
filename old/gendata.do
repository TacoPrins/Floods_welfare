# delimit;
set more 1;
*----------------------------------------;
clear all;
set mem 100m;
set logtype text;
*----------------------------------------;

*READ SAMPLE;
use "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_final.dta";

*DROP TOP PERCENTILES FOR NETWORTH;
/*
foreach t of numlist 1989(3)2007{;
	qui _pctile networth if year ==`t' [fweight = fwgt], p(01 95);
	drop if networth>=r(r2) & year == `t';
};
*/
*AGE RESTRICTIONS;
keep if age>=20 & age<=79;

*GENERATE LABOR INCOME FOR WORKERS;
gen incomenoselfy = income - hh_selfy;      *all income minus self employment y;
drop if labinc<0;
gen labincwork = .;
replace labincwork = labinc; 

*-----------------;
*DEFINITIONS;
*-----------------;
gen cashfrac    = (69*2/2720); *(median $69 per indiv & median liqpos without cash imputation is 2720); 
gen liqpos      = liq*(1+cashfrac); *adjusted by cash holdings (median 69 per indiv);
gen liqposnoret = call + checking + cashfrac*liq;
gen liqneg      = ccdebt;
gen direct      = nmmf + stocks + bond;
gen direct_eq   = deq;
gen direct_bond = direct - deq;
gen housepos    = houses + oresre + nnresre;
gen houseneg    = mrthel + resdbt;
gen nethouse    = housepos - houseneg;
gen netcars     = vehic;
gen sb          = savbnd;
gen certdep     = cds;
gen retacc      = retqliq;
gen retacc_eq   = reteq;
gen lifeins     = cashli;
gen netbus      = 0;
*gen netbus     = bus;

gen othassets   = othfin + othma + othnfin; *EXCLUDE because we also exclude other debt;

gen netliq      = liqpos - liqneg;
gen illiqpos    = housepos + netcars + direct + sb + certdep + retacc + lifeins + netbus; 
gen illiqneg    = houseneg;
gen netilliq    = illiqpos - illiqneg;
gen networthscf = networth - netbus;

gen brliqpos      = liqpos + direct;
gen brliqneg      = ccdebt;
gen netbrliq      = brliqpos - brliqneg;
gen brilliqpos    = housepos + netcars + sb + certdep + retacc + lifeins + netbus; 
gen brilliqneg    = houseneg;
gen netbrilliq    = brilliqpos - brilliqneg;
gen netbrilliqnc  = brilliqpos - brilliqneg - netcars;
gen networthnew   = netbrilliq + netbrliq;
gen networthnc    = netbrilliqnc + netbrliq;

save "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_defs.dta", replace;

// TABLE 3 
sum labincwork networthnc netbrliq liqpos direct liqneg netbrilliqnc nethouse retacc lifeins cd sb [fweight=fwgt],d; 


