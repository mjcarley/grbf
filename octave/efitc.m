function [E,A]=efitc(N, al)

  X = 0:N ;
  A = zeros(N+1, N+1) ;
  for i=0:N
    ##A(i+1,:) = (exp(-al^2*(X-X(i+1)).^2) + exp(-al^2*(X+X(i+1)).^2))*2 ;
    A(i+1,:) = (exp(-al^2*(X-i).^2) + exp(-al^2*(X+i).^2))*2 ;
  endfor
  A(:,1) /= 2 ;
  A(1,:) /= 2 ;
  
  cond(A) 

  ##A(find(abs(A)<1e-16)) = 0 ;
  
  cond(A) 

  b = zeros(N+1, 1) ; b(1) = 1 ;

  E = A\b ;
  #[U,S,V] = svd(A) ;
  #S = diag(S) ;
  
  #E = tikhonov(U, S, V, b, 1e-9) ;
  ##E /= 2 ;
  E = [flipud(E); E(2:end)] ;
  return ;
  
  nd = 12 ;
  Ad = 0*A ;
  for i=-nd:nd
    nr = N - abs(i) + 1 ;
    c0 = 1 ; r0 = -i ;
    if ( i == 0 ) r0 = 1 ; c0 = 1 ; endif
    if ( i < 0 ) r0 = -i+1 ; c0 = 1 ; endif ;
    if ( i > 0 ) r0 =    1 ; c0 = i+1 ; endif ;
    Ad(r0:r0+nr-1, c0:c0+nr-1) += diag(diag(A,i),nr,nr) ;
    
  endfor
  cond(Ad)
  A = inv(Ad)*A ; b = inv(Ad)*b ;
  cond(A) 
  E = A\b ;

  E = [flipud(E); E(2:end)] ;
  
  return ;
  
  X = -N:N ;
  A = zeros(2*N+1, 2*N+1) ;

  for i=1:2*N+1
    A(i,:) = exp(-al^2*(X-X(i)).^2) ;
  endfor

  cond(A) ;
  
  b = zeros(2*N+1, 1) ; b(N+1) = 1 ;

  E = A\b ;
  
  
