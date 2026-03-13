use "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_defs.dta", clear

keep if age>=21 & age<24 & labincwork>=5.15*46*30 & labincwork!=. //drop if hh labour income is less than statutory hourly minimum wage times 46 weeks times 30hrs/week (we have filtered the data to ensure this is the minimum hrs)
gen ln_labincwork=ln(labincwork)
sum ln_labincwork [fweight=fwgt], det
scalar med_lninc = r(p50)
scalar sd_lninc  = r(sd)
*Normalise by sd from median to get normalised coefficients on lab income
replace ln_labincwork=(ln_labincwork-med_lninc)/sd_lninc

*Find coefficients for probability of no wealth
*SET MINIMUM WEALTH THRESHOLD TO AVOID PROBLEM OF EXCESS VARIANCE IN LOG WEALTH
keep if networthscf!=.
replace networthscf=0 if networthscf<1000
gen nowealth=0
replace nowealth=1 if networthscf<=0  
logit nowealth ln_labincwork [fweight=fwgt]

*Find variance of wealth and correlation with lab income conditional on having positive wealth
gen ln_networthscf=ln(networthscf) if networthscf>0
sum ln_networthscf   [fweight=fwgt] if ln_networthscf!=., det
corr ln_networthscf ln_labincwork [fweight=fwgt] if ln_networthscf!=.

*Distribution of wealth scaled by median inc 
gen networth_normalised=networthscf/exp(med_lninc) 
sum networth_normalised [fweight=fwgt] , det

*Find the age profile for income
use "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_defs.dta", clear
keep if age>=21 & age<66 & labincwork>=5.15*46*30 & labincwork!=. & age!=.
gen ln_labincwork=ln(labincwork)
gen mean_age_lninc=0
forvalues a = 21(1)65 {
    quietly summarize ln_labincwork [fweight=fwgt] if age==`a'
    replace mean_age_lninc = r(mean) if age==`a'
	di r(mean)
}
*replace age=(age-21)/3

gen age2 = age^2
gen age3 = age^3
gen age4 = age^4
reg mean_age_lninc age age2 age3 age4 [fweight=fwgt]

forvalues a = 21(1)65 {
    di `a'
	di _b[_cons]+_b[age]*`a'+_b[age2]*`a'^2+_b[age3]*`a'^3+_b[age4]*`a'^4
}

*Generate statistics for end-of-life wealth inequality 
use "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_defs.dta", clear
keep if age>=75 & age<81 & age!=. & networthscf!=.
sum networthscf   [fweight=fwgt], det

_pctile networthscf [fweight=fwgt], p(33 67)
di r(r2)/r(r1)

gen financial_wealth=networthscf - nethouse
keep if financial_wealth!=.
sum financial_wealth [fweight=fwgt], det
_pctile financial_wealth [fweight=fwgt], p(33 67)
di r(r2)/r(r1)

cumul financial_wealth [fweight=fwgt], gen(cdf_finwealth)
line cdf_finwealth financial_wealth if financial_wealth<1000000, sort