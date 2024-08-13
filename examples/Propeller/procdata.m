## convenience script to process propeller thrust and torque data
## propeller diameter and speed
D = 10*25.4/1000 ;
N = 6700/60 ;

## experimental data
dat = load("6700RPM.txt") ;
Je = dat(:,1) ; CTe = dat(:,2) ; CPe = dat(:,3) ;

## output from LLT
load pcurve.dat

U = pcurve(:,1) ;
T = pcurve(:,2);
Q = pcurve(:,3) ;
J = U./(N*D) ;
CT = T./N^2/D.^4 ;
## negative because torque on propeller is opposite of input torque
CP = -2*pi*Q/N^2/D^5 ;
