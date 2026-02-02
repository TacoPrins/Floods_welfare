# delimit;
set more 1;
*----------------------------------------;
clear all;
set mem 100m;
set logtype text;

*----------------------------------------;
* READ IN NEW VARIABLES;
* DESCRIPTION;
*----------------------------------------;

*HAVE CREDIT OR CHARGE CARDS;
*------------------------------;
*HASCC=X410: 1=YES, 5=NO;
*HASMCVISA=X7973: 0=NA (no cc), 1=YES, 5=NO;
*HASAMEX=X7976: 0=NA (no cc), 1=YES, 5=NO;

*REVOLVING DEBT VARIABLES;
*------------------------------;
*REVBALANCE1=X413: -1=NONE, 0=NA;
*REVBALANCE2=X421: -1=NONE, 0=NA;
*REVBALANCE3=X424: -1=NONE, 0=NA;
*REVBALANCE4=X427: -1=NONE, 0=NA;

*CREDIT LIMIT AND INTRATE VARIABLES;
*------------------------------;
*MAXCREDIT=X414: -1:NONE, 0=NA;
*INTRATECC=X7132: -1=NO INTEREST 0=NA;

*PAY FREQUENCY;
*-------------------------------;
*PAYFREQ=X432: 1=ALWAYS, 3=SOMETIMES, 5=NEVER, 0=NA;

*NO ACCESS TO CREDIT;
*-------------------------------;
*TURNEDOWN=X407: 1 OR 3=YES;
*NOTAPPLIED X409: 1=YES;

*INCOME VARIABLES
*------------------------------;
*HH_EARNINGS=X5702: 0=NO EARNINGS;
*SELFY = X5704: 0, -1, -2: NO EARNINGS
*UIBEN=X5716: 0=NA;
*CHILDBEN=X5718: 0=NA;
*TANF=X5720: 0=NA;
*SSINC=X5722: 0=NA;
*OTHINC=X5724: -1:NONE, 0=NA;
*SOURCE_OTHINC=x5725;

*PENSION PLANS VARIABLES;
*-----------------------------;
*ENROLLED IN EMPLOYER PLAN=X4135 (HEAD) & X4735 (WIFE): 1=yes
*FRACTION OF EARNINGS CONTRIB =X4206 & 4806: -1=nothing -2:not valid 0: NA
*FRACTION OF EARNINGS EMPLOYERS CONTRIB = X4219 & 4819: -1=nothing -2:not valid 0: NA;

*SAVE NOTHING;
*-----------------------------;
*DON'T SAVE - USUALLY SPEND MORE THAN INCOME = X3015: 1=YES ;
*DON'T SAVE - USUALLY SPEND ABOUT AS MUCH AS INCOME = X3016: 1=YES   ; 
*SAVE WHATEVER IS LEFT = X3017: 1=YES   ; 
*SPENDING EXCEEDED INCOME = X7510: 1 OR 2=YES;
*BOUGHT CAR OR HOUSE = X7509: 1=YES 5=NO;

*FREQUENCY OF PAYMENT;
*------------------------------;
*EARNINGS ON MAIN JOB(SPOUSE 1) = X4112: -1=NONE, 0=NA
*EARNINGS ON MAIN JOB(SPOUSE 2) = X4712: -1=NONE, 0=NA
*FREQUENCY OF PAYMENT (SPOUSE 1) = X4113: <=0: NA
*FREQUENCY OF PAYMENT (SPOUSE 2) = X4713: <=0: NA
*EARNINGS ON MAIN JOB(SPOUSE 1 -SELFEMP) = X4131: -1=NONE, 0=NA
*EARNINGS ON MAIN JOB(SPOUSE 2 -SELFEMP) = X4731: -1=NONE, 0=NA
*FREQUENCY OF PAYMENT (SPOUSE 1 -SELFEMP) = X4132: <=0: NA
*FREQUENCY OF PAYMENT (SPOUSE 2 -SELFEMP) = X4732: <=0: NA

*------------------------------;
*COLLECT THE NEW VARIABLES;
*------------------------------;
use y1 x432 x413 x421 x424 x427 x7132
       x410 x7973 x7976
	   x816
       x414 x407 x409 
	   x5702 x5704 x5716 x5718 x5720 x5722 x5724 x5725
	   x4135 x4206 x4219 x4735 x4806 x4819
	   x3015 x3016 x3017
	   x7509 x7510
	   x4110 x4111 x4112 x4113 x4710 x4711 x4712 x4713 x4131 x4132 x4731 x4732
	   
	   using "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001.dta";

keep if (x4110>=30&x4111>=46)|(x4710>=30&x4711>=46); //NEW CODE: Require min hrs of work per week and weeks per year
rename x410 hascc;
rename x7973 hasmcvisa;
rename x7976 hasamex;
replace hascc=0 if (hascc==5 | (hasmcvisa==5 & hasamex==5));  
rename x432  payfreq;
rename x413 revbalance1; 
rename x421 revbalance2; 
rename x424 revbalance3; 
rename x427 revbalance4;
rename x7132 intratecc;
replace intratecc=0 if intratecc==-1;
replace revbalance1=0 if revbalance1==-1;
replace revbalance2=0 if revbalance2==-1;
replace revbalance3=0 if revbalance3==-1;
replace revbalance4=0 if revbalance4==-1;
gen ccdebt=0;
replace ccdebt = (revbalance1+revbalance2 +revbalance3 +revbalance4) if payfreq>1;

rename x816 intratemm;
replace intratemm=0 if intratemm==-1;

rename x414  maxcredit;
rename x407  turnedown;
rename x409  notapplied;
replace maxcredit=0 if maxcredit==-1;
gen credenied=0;
replace credenied=1 if turnedown==1 |turnedown==3;

rename x5702 hh_earnings;
rename x5704 hh_selfy;
rename x5716 uiben;
rename x5718 childben;
rename x5720 tanf;
rename x5722 ssinc;
rename x5724 source_othinc;
rename x5725 othinc;
replace othinc=0 if othinc==-1 | source_othinc ==11 | source_othinc ==14 | source_othinc ==30 | source_othinc ==36;
gen labinc = hh_earnings+uiben+childben+tanf+ssinc+othinc;
gen labincplus = hh_earnings+uiben+childben+tanf+ssinc+othinc+hh_selfy;

rename x4135 hhasemplpens;
replace hhasemplpens = 0 if hhasemplpens~=1;
rename x4206 hcontrib;
replace hcontrib =0 if hcontrib==-1 | hcontrib ==-2;
rename  x4219 hemplcontrib;
replace hemplcontrib =0 if hemplcontrib==-1 | hemplcontrib ==-2;

rename x4735 whasemplpens;
replace whasemplpens = 0 if whasemplpens~=1;
rename x4806 wcontrib;
replace wcontrib =0 if wcontrib==-1 | wcontrib ==-2;
rename  x4819 wemplcontrib;
replace wemplcontrib =0 if wemplcontrib==-1 | wemplcontrib ==-2;

gen famhasemplpens=0;
replace famhasemplpens=1 if hhasemplpens==1 | whasemplpens==1;

rename x3015 nosavebor;
rename x3016 nosavezero;
rename x3017 savewhatta;
gen htm1=0;
replace htm1 = 1 if nosavebor==1 | nosavezero==1 ;
gen htm2=0;
replace htm2=1 if nosavebor==1 | nosavezero==1 | savewhatta==1;

rename x7510 spendmorey;
rename x7509 buyhome;
gen htm3=0;
replace htm3=1 if (spendmorey==1 | spendmorey==2) & buyhome==5;
gen htm4=0;
replace htm4=1 if (spendmorey==1 | spendmorey==2);
  
rename x4112 labearn1; 
rename x4113 freqle1; 
rename x4712 labearn2; 
rename x4713 freqle2; 
rename x4131 selfearn1;
rename x4132 freqse1; 
rename x4731 selfearn2;
rename x4732 freqse2; 
replace labearn1 = 0 if labearn1<=0;
replace labearn2 = 0 if labearn2<=0;
replace selfearn1 = 0 if selfearn1<=0;
replace selfearn2 = 0 if selfearn2<=0;
gen hh_earningsalt = labearn1+labearn2+selfearn1+selfearn2;

sort y1;
save "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_moredata.dta", replace;

*MERGE WITH REST OF THE 2001 DATA;
use "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scfgv.dta";
sort y1;
keep if year==2001;

merge y1 using "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_moredata.dta";
tab _merge;
drop if _merge==2;
drop _merge;
drop x*;

*WEIGHT;
gen fwgt = round(wgt);

save "C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\kaplan_violante_ecmtra_2014_replicationmaterials (1)\ReplicationFiles\SCF\data\scf2001_final.dta", replace;