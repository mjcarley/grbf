b  = 60.62*25.4/1000 ;
##Om = 87.965 ;
Om=62.895
al0 = 0.0695 ;
da  = 0.0724 ;
T = 0.4 ;
T0 = 0.0 ;
nt = 255 ;
##T = 1 ;
##Om = 7 ;
##da = 0.08 ;
t = T0 + (1:nt)/nt*(T-T0) ;
th = al0 + da*sin(Om*t) ;
stub = "Data/solution-%d.dat" ;

y = -b/2 + b*[0.25 0.475 0.8 0.966] ;

cl = [] ; al = [] ; u = v = [] ;
for i=1:nt
  fname = sprintf(stub, i) ;
  dat = load(fname) ;
  ci = interp1(dat(:,2),dat(:,4),y) ;
  ##ci = dat(1:4,4)' ;
  cl = [cl; ci] ;
  v1 = interp1(dat(:,2),dat(:,6),y) ;
  v2 = interp1(dat(:,2),dat(:,7),y) ;
  ##v1 = dat(1:4,6)' ;
  ##v2 = dat(1:4,7)' ;
  u = [u; v1] ;
  v = [v; v2] ;
  al = [al; atan2(v2, v1)] ;
endfor
