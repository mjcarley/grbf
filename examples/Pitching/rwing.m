## generate a basic rectangular wing for NASA TR-4632
ch = 12*25.4/1000 ;
b  = 60.62*25.4/1000 ;

np = 20 ;

yb = linspace(-b/2,b/2,np+1)' ;
tb = linspace(pi,2*pi,np+1)' ;
yb = b/2*cos(tb) ;
xb = [0*yb yb 0*yb] ;
xb(:,1) 
xw = xb  ;
xw(:,1) += 0.75*ch ;

tc = tb(1:end-1) + 0.5*diff(tb(1:2)) ;
##xc = xb(1:end-1,:) + 0.5*diff(xb) ;
yc = b/2*cos(tc) ;
xc = [0*yc yc 0*yc] ;
nc = xc*0 ; nc(:,3) = 1 ;
cc = xc*0 ; cc(:,1) = 1 ;
chc = ch*ones(size(xc(:,1))) ;

fid = fopen("wing.llt", "w") ;

fprintf(fid, "LINE %d\n", np) ;

dat = [xc nc cc chc] ;
fprintf(fid, "%e %e %e %e %e %e %e %e %e %e\n", dat') ;
dat = [xb xw] ;
fprintf(fid, "%e %e %e %e %e %e\n", dat') ;

fclose(fid) ;
