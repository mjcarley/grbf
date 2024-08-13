## generate a basic elliptical wing

## semi root chord and span
a = 0.5 ; b = 2.5 ;
AR = 4*b/pi/a ;

np = 40 ;

yb = linspace(-b,b,np+1)' ;
##yb = yb(2:end-1) ;

x = a*[-sqrt(1-yb.^2/b^2) sqrt(1-yb.^2/b^2)] ;

## chord lengths
ch = diff(x, [], 2) ;
## quarter chord points
xq = x(:,1) + 0.25*ch ;

xb = [0*yb yb 0*yb] ;

##xb(:,1) += 0.12*ch ;

xw = xb  ;
xw(:,1) += 0.75*ch ;

xc = xb(1:end-1,:) + 0.5*diff(xb) ;
nc = xc*0 ; nc(:,3) = 1 ;
cc = xc*0 ; cc(:,1) = 1 ;
chc = 2*a*sqrt(1-xc(:,2).^2/b^2) ;

fid = fopen("wing.llt", "w") ;

fprintf(fid, "LINE %d\n", np) ;

dat = [xc nc cc chc] ;
fprintf(fid, "%e %e %e %e %e %e %e %e %e %e\n", dat') ;
dat = [xb xw] ;
fprintf(fid, "%e %e %e %e %e %e\n", dat') ;

fclose(fid) ;
