/* This file is part of GRBF, a library for Radial Basis Function
 * interpolation using Gaussians.
 *
 * Copyright (C) 2024 Michael Carley
 *
 * GRBF is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. GRBF is distributed in the
 * hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GRBF.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <stdio.h>
#include <math.h>

#include <glib.h>

#include <blaswrap.h>

#include <grbf.h>

#include "grbf-private.h"

/**
 *
 * @ingroup cardinal
 * 
 * @{
 * 
 */

/** 
 * Evaluate the coefficients \f$E_{i}\f$ of the cardinal function for
 * Gaussian RBFs, \f$f(x)=\sum E_{i}\exp[-\alpha^{2}(x-i)^{2}]\f$,
 * \f$f(0)=1\f$, \f$f(x)=0\f$, \f$x=\pm1, \pm2, \ldots\f$, defined by
 * Boyd and Wang, 2009, https://dx.doi.org/10.1016/j.amc.2009.08.037.
 * 
 * @param al overlap parameter \f$\alpha\f$;
 * @param N number of coefficients to evaluate;
 * @param E on exit contains coefficients;
 * @param duplicate if TRUE, return symmetric terms \f$E_{-i}\f$, 
 * otherwise only return coefficients for \f$i\geq0\f$;
 * @param rcond if not \c NULL, on exit contains reciprocal condition
 * number of fitting matrix (useful for error estimation);
 * @param work workspace with \f$(N+1)^{2}\f$ elements or 
 * \f$(N+1)^2+3N+1\f$ elements if \a rcond is not \c NULL.
 * 
 * @return 0 on success.
 */

gint grbf_cardinal_function_coefficients(gdouble al, gint N,
					 gdouble *E, gboolean duplicate,
					 gdouble *rcond, gdouble *work)

{
  gdouble *Es, *A, anorm ;
  gint n, info, i1 = 1, i, j, iwork[1024] ;
  
  if ( duplicate ) Es = &(E[N]) ; else Es = E ;

  A = work ; g_assert(N < 1023) ;

  n = N + 1 ;  
  for ( i = 0 ; i < n ; i ++ ) {
    for ( j = i ; j < n ; j ++ ) {
      /*FORTRAN ordering (but the matrix is symmetric ...)*/
      A[i*n+j] = A[j*n+i] =
	2.0*(exp(-al*al*(j-i)*(j-i)) + exp(-al*al*(j+i)*(j+i))) ;
    }
  }
  for ( i = 0 ; i < n ; i ++ ) {
    A[i*n+0] *= 0.5 ; A[0*n+i] *= 0.5 ;
    Es[i] = 0.0 ;
  }
  Es[0] = 1.0 ;

  if ( rcond != NULL ) {
    anorm = dlansy_("1", "U", &n, A, &n, &(A[n*n])) ;
  }
  
  dpotrf_("U", &n, A, &n, &info) ;

  if ( rcond != NULL ) {
    dpocon_("U", &n, A, &n, &anorm, rcond, &(A[n*n]), iwork, &info) ;
  }

  dpotrs_("U", &n, &i1, A, &n, Es, &n, &info) ;

  if ( !duplicate ) return 0 ;

  for ( i = 0 ; i < N ; i ++ ) {
    E[i] = E[2*N-i] ;
  }
  
  return 0 ;
}

/** 
 * Evaluate the cardinal function for Gaussian RBFs \f$f(x)=\sum
 * E_{i}\exp[-\alpha^{2}(x-i)^{2}]\f$, \f$f(0)=1\f$, \f$f(x)=0\f$,
 * \f$x=\pm1, \pm2, \ldots\f$, defined by Boyd and Wang, 2009,
 * https://dx.doi.org/10.1016/j.amc.2009.08.037.
 * 
 * @param al overlap parameter \f$\alpha\f$; 
 * @param x evaluation point \f$x\f$.
 * 
 * @return \f$f(x)\f$.
 */

gdouble grbf_cardinal_func(gdouble al, gdouble x)

{
  gdouble C ;
  
  if ( x == 0 ) return 1.0 ;

  C = al*al*sin(M_PI*x)/M_PI/sinh(al*al*x) ;
  
  return C ;
}

gint grbf_cardinal_interpolation_eval_1d(gdouble al,
					 gdouble *F, gint fstr, gint nf,
					 gint nc,
					 gdouble x, gdouble *f)

{
  gint i, j ;

  for ( i = 0 ; i < nf ; i ++ ) {
    for ( j = 0 ; j < nc ; j ++ ) 
      f[j] += F[i*fstr+j]*grbf_cardinal_func(al, x-i) ;
  }    
  
  return 0 ;
}

gint grbf_cardinal_interpolation_eval_2d(gdouble al,
					 gdouble *F, gint fstr,
					 gint ni, gint ldf, gint nj,
					 gint nc,
					 gdouble *x, gdouble *f)

{
  gint i, j, k ;
  gdouble Cx, Cy ;
  
  g_assert(ni <= ldf) ;

  for ( i = 0 ; i < ni ; i ++ ) {
    Cx = grbf_cardinal_func(al, x[0]-i) ;
    for ( j = 0 ; j < nj ; j ++ ) {
      Cy = grbf_cardinal_func(al, x[1]-j) ;      
      for ( k = 0 ; k < nc ; k ++ ) {
	f[k] += F[i*ldf+j*fstr+k]*Cx*Cy ;
      }
    }
  }
  
  return 0 ;
}

gint grbf_cardinal_interpolation_eval_3d(gdouble al,
					 gdouble *F, gint fstr,
					 gint ni, gint nj, gint nk,
					 gint nc,
					 gdouble *x, gdouble *f)

{
  gint i, j, k, n ;
  gdouble Cx, Cy, Cz ;
  
  for ( i = 0 ; i < ni ; i ++ ) {
    Cx = grbf_cardinal_func(al, x[0]-i) ;
    for ( j = 0 ; j < nj ; j ++ ) {
      Cy = grbf_cardinal_func(al, x[1]-j) ;      
      for ( k = 0 ; k < nk ; k ++ ) {
	Cz = grbf_cardinal_func(al, x[2]-k) ;
	for ( n = 0 ; n < nc ; n ++ ) {
	  f[n] += F[(i*nj*nk+j*nk+k)*fstr+n]*Cx*Cy*Cz ;
	}
      }
    }
  }
  
  return 0 ;
}

/**
 *
 * @}
 *
 */
