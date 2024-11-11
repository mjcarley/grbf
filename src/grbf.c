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
 * @ingroup weights
 * 
 * @{
 * 
 */

gint GRBF_FUNCTION_NAME(grbf_interpolation_weights_1d_slow)(GRBF_REAL *F,
							    gint fstr,
							    gint nf, gint nc,
							    GRBF_REAL *E,
							    gint N,
							    gboolean duplicated,
							    GRBF_REAL *w,
							    gint wstr)

{
  gint i ;

  g_assert(fstr >= nc) ;
  g_assert(wstr >= nc) ;
  
  for ( i = 0 ; i < nf ; i ++ ) {
    GRBF_FUNCTION_NAME(grbf_interpolation_weight_1d)(F, fstr, nf, nc, E, N,
						     duplicated, i,
						     &(w[i*wstr])) ;
  }

  return 0 ;
}

gint GRBF_FUNCTION_NAME(grbf_interpolation_weight_1d)(GRBF_REAL *F, gint fstr,
						      gint nf, gint nc,
						      GRBF_REAL *E, gint N,
						      gboolean duplicated,
						      gint i, GRBF_REAL *w)
{
  gint j, k, n, i1 = 1 ;
  
  g_assert(duplicated) ;

  j = MAX(-N,-i) ; n = MIN(nf-i,N) - j + 1 ;
  for ( k = 0 ; k < nc ; k ++ ) {
#ifdef GRBF_SINGLE_PRECISION
    w[k] = blaswrap_sdot(n,&(F[(i+j)*fstr+k]),fstr,&(E[j+N]),i1) ;
#else /*GRBF_SINGLE_PRECISION*/
    w[k] = blaswrap_ddot(n,&(F[(i+j)*fstr+k]),fstr,&(E[j+N]),i1) ;
#endif /*GRBF_SINGLE_PRECISION*/
  }

  return 0 ;
}

static gint interpolation_weights_2d_unit(GRBF_REAL *F,
					  gint ni, gint ldf, gint nj,
					  GRBF_REAL *E, gint N,
					  gboolean duplicated,
					  GRBF_REAL *w, gint wstr, gint ldw)

{
  gint i, j, i1, j1 ;
  GRBF_REAL work[2048] ;

  /*convolution using separable property of E*/
  /*https://en.wikipedia.org/wiki/Multidimensional_discrete_convolution#Row-column_decomposition*/
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( i1 = MAX(-N,-i) ; i1 <= MIN(ni-i,N) ; i1 ++ ) {
      for ( j = 0 ; j < nj ; j ++ ) {
	w[(i*ldw+j)*wstr] += F[(i+i1)*ldf+j]*E[i1+N] ;
      }
    }
  }
  /*now convolve columns*/
  for ( i = 0 ; i < ni ; i ++ ) {
    /*copy column into work for processing*/
    for ( j = 0 ; j < nj ; j ++ ) {
      work[j] = w[(i*ldw+j)*wstr] ;
      w[(i*ldw+j)*wstr] = 0 ;
    }
    /*and convolve back into place*/
    for ( j = 0 ; j < nj ; j ++ ) {
      for ( j1 = MAX(-N,-j) ; j1 <= MIN(nj-j,N) ; j1 ++ ) {
	w[(i*ldw+j)*wstr] += work[j+j1]*E[j1+N] ;
      }
    }
  }
  
  return 0 ;
}

gint GRBF_FUNCTION_NAME(grbf_interpolation_weights_2d_slow)(GRBF_REAL *F,
							    gint fstr, gint nc,
							    gint ni, gint ldf,
							    gint nj,
							    GRBF_REAL *E,
							    gint N,
							    gboolean duplicated,
							    GRBF_REAL *w,
							    gint wstr, gint ldw)

{
  gint i, i1, j, j1, n, k, one = 1 ;
  
  g_assert(nc > 0) ;
  g_assert(nj <= ldf) ;
  g_assert(duplicated) ;
  g_assert(fstr >= nc) ;
  g_assert(wstr >= nc) ;

#if 1
  if ( fstr == 1 ) {
    return interpolation_weights_2d_unit(F, ni, ldf, nj, E, N, duplicated,
					 w, wstr, ldw) ;
  }
  
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( i1 = MAX(-N,-i) ; i1 <= MIN(ni-i,N) ; i1 ++ ) {
      for ( j = 0 ; j < nj ; j ++ ) {
	j1 = MAX(-N,-j) ; n = MIN(N,nj-j) - j1 + 1 ;
	for ( k = 0 ; k < nc ; k ++ ) {
#ifdef GRBF_SINGLE_PRECISION
	  w[i*ldw+j*wstr+k] +=
	    E[i1+N]*blaswrap_sdot(n,&(F[(i+i1)*ldf+(j+j1)*fstr+k]),fstr,
				  &(E[j1+N]),one) ;
#else /*GRBF_SINGLE_PRECISION*/
	  w[i*ldw+j*wstr+k] +=
	    E[i1+N]*blaswrap_ddot(n,&(F[(i+i1)*ldf+(j+j1)*fstr+k]),fstr,
				  &(E[j1+N]),one) ;
#endif /*GRBF_SINGLE_PRECISION*/
	}
      }
    }
  }

#else
  /*this is the slow version so I can remember how I did this*/
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( i1 = MAX(-N,-i) ; i1 <= MIN(ni-i,N) ; i1 ++ ) {
      for ( j = 0 ; j < nj ; j ++ ) {
	for ( j1 = MAX(-N,-j) ; j1 <= MIN(nj-j,N) ; j1 ++ ) {
	  for ( k = 0 ; k < nc ; k ++ ) {
	    w[(i*nj+j)*wstr+k] +=
	      F[((i+i1)*ldf+(j+j1))*fstr+k]*E[i1+N]*E[j1+N] ;
	  }
	}
      }
    }
  }
#endif
  
  return 0 ;
}

gint GRBF_FUNCTION_NAME(grbf_interpolation_weights_3d)(GRBF_REAL *F, gint fstr,
						       gint nc,
						       gint ni, gint nj,
						       gint nk,
						       GRBF_REAL *E, gint N,
						       gboolean duplicated,
						       GRBF_REAL *w, gint wstr)

{
  gint i, i1, j, j1, k, k1, m ;
  
  g_assert(nc > 0) ;
  g_assert(duplicated) ;
  g_assert(fstr >= nc) ;
  g_assert(wstr >= nc) ;

  /*this is the slow version so I can remember how I did this*/
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( i1 = MAX(-N,-i) ; i1 <= MIN(ni-i,N) ; i1 ++ ) {
      for ( j = 0 ; j < nj ; j ++ ) {
	for ( j1 = MAX(-N,-j) ; j1 <= MIN(nj-j,N) ; j1 ++ ) {
	  for ( k = 0 ; k < nk ; k ++ ) {
	    for ( k1 = MAX(-N,-k) ; k1 <= MIN(nk-k,N) ; k1 ++ ) {
	      for ( m = 0 ; m < nc ; m ++ ) {
		w[(i*nj*nk+j*nk+k)*wstr+m] +=
		  F[((i+i1)*nj*nk+(j+j1)*nk+k+k1)*fstr+m]*
		  E[i1+N]*E[j1+N]*E[k1+N] ;
	      }
	    }
	  }
	}
      }
    }
  }
  
  return 0 ;
}

/** end of weights documentation group**/

/**
 *
 * @}
 *
 */

/**
 *
 * @ingroup interp
 * 
 * @{
 * 
 */

/** 
 * Evaluate an interpolant based on a unit grid.
 * 
 * @param al overlap parameter \f$\alpha\f$;
 * @param w interpolation weights, evaluated using 
 * ::grbf_interpolation_weights_1d_slow or
 * ::grbf_interpolation_weights_1d_fft;
 * @param wstr data stride in \a w;
 * @param nw number of data points in \a w;
 * @param x evaluation coordinate;
 * @param f on output contains interpolated value.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_interpolation_eval_1d)(GRBF_REAL al,
						    GRBF_REAL *w, gint wstr,
						    gint nw,
						    GRBF_REAL x, GRBF_REAL *f)

{
  gint i ;

  for ( i = 0 ; i < nw ; i ++ ) {
    (*f) += w[i*wstr]*exp(-al*al*(x-i)*(x-i)) ;
  }
  
  return 0 ;
}

/** 
 * Evaluate a Gaussian RBF interpolant with arbitrarily spaced
 * interpolation nodes, \f$\mathbf{f}=\sum
 * \mathbf{w}_{i}\exp[-(x-y_{i})^{2}/\sigma_{i}^{2}]\f$,
 * in one dimension.
 * 
 * @param y interpolation node coordinates;
 * @param ystr stride in \a y;
 * @param ny number of interpolation nodes;
 * @param s interpolation node Gaussian parameters, \f$\sigma_{i}\f$;
 * @param sstr stride in \a s (can be zero to set constant \f$\sigma\f$); 
 * @param w interpolation weights;
 * @param wstr stride in \a w;
 * @param nc number of components in interpolated function;
 * @param x coordinate of evaluation node;
 * @param f on exit contains interpolated function.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_gaussian_eval_1d)(GRBF_REAL *y, gint ystr, gint ny,
					       GRBF_REAL *s, gint sstr,
					       GRBF_REAL *w, gint wstr, gint nc,
					       GRBF_REAL  x, GRBF_REAL *f)

{
  gint i, j ;
  GRBF_REAL R2, s2 ;
  
  for ( i = 0 ; i < ny ; i ++ ) {
    R2 = grbf_vector1d_distance2(&x,&(y[i*ystr])) ;
    for ( j = 0 ; j < nc ; j ++ ) {
      s2 = s[i*sstr]*s[i*sstr] ;
      f[j] += w[i*wstr+j]*exp(-R2/s2) ;
    }
  }
  
  return 0 ;
}

/** 
 * Evaluate a Gaussian RBF interpolant with arbitrarily spaced
 * interpolation nodes, \f$\mathbf{f}=\sum
 * \mathbf{w}_{i}\exp[-|\mathbf{x}-\mathbf{y}_{i}|^{2}/\sigma_{i}^{2}]\f$,
 * in two dimensions.
 * 
 * @param y interpolation node coordinates;
 * @param ystr stride in \a y;
 * @param ny number of interpolation nodes;
 * @param s interpolation node Gaussian parameters, \f$\sigma_{i}\f$;
 * @param sstr stride in \a s (can be zero to set constant \f$\sigma\f$); 
 * @param w interpolation weights;
 * @param wstr stride in \a w;
 * @param nc number of components in interpolated function;
 * @param x coordinate of evaluation node;
 * @param f on exit contains interpolated function.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_gaussian_eval_2d)(GRBF_REAL *y, gint ystr, gint ny,
					       GRBF_REAL *s, gint sstr,
					       GRBF_REAL *w, gint wstr, gint nc,
					       GRBF_REAL *x, GRBF_REAL *f)

{
  gint i, j ;
  GRBF_REAL R2, s2 ;

  g_assert(ystr > 1) ;
  g_assert(wstr >= nc) ;
  
  for ( i = 0 ; i < ny ; i ++ ) {
    R2 = grbf_vector2d_distance2(x,&(y[i*ystr])) ;
    for ( j = 0 ; j < nc ; j ++ ) {
      s2 = s[i*sstr]*s[i*sstr] ;
      f[j] += w[i*wstr+j]*exp(-R2/s2) ;
    }
  }
  
  return 0 ;
}

gint GRBF_FUNCTION_NAME(grbf_interpolation_eval_2d)(GRBF_REAL al,
						    GRBF_REAL *w, gint wstr,
						    gint nw,
						    gint ni, gint ldw, gint nj,
						    GRBF_REAL *x, GRBF_REAL *f)

{
  gint i, j, k, i0, j0, di, dj ;
  GRBF_REAL R2, tol ;

  tol = 1e-12 ;
  R2 = -log(tol)/al/al ;

  di = (gint)ceil(sqrt(R2)) + 1 ;
  dj = di ;

  i0 = (gint)floor(x[0]) ;
  j0 = (gint)floor(x[1]) ;

  for ( i = MAX(0,i0-di) ; i < MIN(ni,i0+di) ; i ++ ) {
    for ( j = MAX(0,j0-dj) ; j < MIN(nj,j0+dj) ; j ++ ) {
      R2 = (x[0]-i)*(x[0]-i) + (x[1]-j)*(x[1]-j) ;
      for ( k = 0 ; k < nw ; k ++ ) {
	f[k] += w[i*ldw+j*wstr+k]*exp(-al*al*R2) ;
      }
    }
  }
  
  return 0 ;
}

/** 
 * Evaluate a Gaussian RBF interpolant with arbitrarily spaced
 * interpolation nodes, \f$\mathbf{f}=\sum
 * \mathbf{w}_{i}\exp[-|\mathbf{x}-\mathbf{y}_{i}|^{2}/\sigma_{i}^{2}]\f$,
 * in three dimensions.
 * 
 * @param y interpolation node coordinates;
 * @param ystr stride in \a y;
 * @param ny number of interpolation nodes;
 * @param s interpolation node Gaussian parameters, \f$\sigma_{i}\f$;
 * @param sstr stride in \a s (can be zero to set constant \f$\sigma\f$); 
 * @param w interpolation weights;
 * @param wstr stride in \a w;
 * @param nc number of components in interpolated function;
 * @param x coordinate of evaluation node;
 * @param f on exit contains interpolated function.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_gaussian_eval_3d)(GRBF_REAL *y, gint ystr, gint ny,
					       GRBF_REAL *s, gint sstr,
					       GRBF_REAL *w, gint wstr, gint nc,
					       GRBF_REAL *x, GRBF_REAL *f)

{
  gint i, j ;
  GRBF_REAL R2, s2 ;

  g_assert(ystr > 2) ;
  g_assert(wstr >= nc) ;
  
  for ( i = 0 ; i < ny ; i ++ ) {
    R2 = grbf_vector3d_distance2(x,&(y[i*ystr])) ;
    for ( j = 0 ; j < nc ; j ++ ) {
      s2 = s[i*sstr]*s[i*sstr] ;
      f[j] += w[i*wstr+j]*exp(-R2/s2) ;
    }
  }
  
  return 0 ;
}

gint GRBF_FUNCTION_NAME(grbf_interpolation_eval_3d)(GRBF_REAL al,
						    GRBF_REAL *w, gint wstr,
						    gint nw,
						    gint ni, gint nj, gint nk,
						    GRBF_REAL *x, GRBF_REAL *f)

{
  gint i, j, k, i0, j0, k0, di, dj, dk, m ;
  GRBF_REAL R2, tol ;

  tol = 1e-12 ;
  R2 = -log(tol)/al/al ;

  di = (gint)ceil(sqrt(R2)) + 1 ;
  dj = di ;
  dk = di ;
  
  i0 = (gint)floor(x[0]) ;
  j0 = (gint)floor(x[1]) ;
  k0 = (gint)floor(x[2]) ;
  
  for ( i = MAX(0,i0-di) ; i < MIN(ni,i0+di) ; i ++ ) {
    for ( j = MAX(0,j0-dj) ; j < MIN(nj,j0+dj) ; j ++ ) {
      for ( k = MAX(0,k0-dk) ; k < MIN(nk,k0+dk) ; k ++ ) {
	R2 = (x[0]-i)*(x[0]-i) + (x[1]-j)*(x[1]-j) + (x[2]-k)*(x[2]-k) ;
	for ( m = 0 ; m < nw ; m ++ ) {
	  f[m] += w[(i*nj*nk+j*nk+k)*wstr+m]*exp(-al*al*R2) ;
	}
      }
    }
  }
  
  return 0 ;
}

static void adjust_limits(GRBF_REAL *xmin, GRBF_REAL *xmax, gint *nx,
			  GRBF_REAL del)

{
  *nx = (gint)(ceil(((*xmax)-(*xmin))/del)) + 1 ;

  *xmax = *xmin + ((*nx)-1)*del ;
  
  return ;
}

gint GRBF_FUNCTION_NAME(grbf_grid_adjust_2d)(GRBF_REAL *xmin, GRBF_REAL *xmax,
					     gint *nx,
					     GRBF_REAL *ymin, GRBF_REAL *ymax,
					     gint *ny,
					     GRBF_REAL del)

{
  adjust_limits(xmin, xmax, nx, del) ;
  adjust_limits(ymin, ymax, ny, del) ;

  return 0 ;
}

gint GRBF_FUNCTION_NAME(grbf_grid_adjust_3d)(GRBF_REAL *xmin, GRBF_REAL *xmax,
					     gint *nx,
					     GRBF_REAL *ymin, GRBF_REAL *ymax,
					     gint *ny,
					     GRBF_REAL *zmin, GRBF_REAL *zmax,
					     gint *nz,
					     GRBF_REAL del)

{
  adjust_limits(xmin, xmax, nx, del) ;
  adjust_limits(ymin, ymax, ny, del) ;
  adjust_limits(zmin, zmax, nz, del) ;
  
  return 0 ;
}

gint GRBF_FUNCTION_NAME(grbf_grid_count_points_3d)(GRBF_REAL *F, gint fstr,
						   gint ni, gint ldfi,
						   gint nj, gint ldfj,
						   gint nk, gint nc,
						   GRBF_REAL tol)

{
  gint i, j, k, m, n ;
  GRBF_REAL norm ;
  
  n = 0 ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      for ( k = 0 ; k < nk ; k ++ ) {
	norm = 0 ;
	for ( m = 0 ; m < nc ; m ++ ) {
	  norm += F[i*ldfi + j*ldfj + k*fstr + m]*
	    F[i*ldfi + j*ldfj + k*fstr + m] ;
	}
	if ( norm > tol*tol ) n ++ ;
      }
    }
  }
  
  return n ;
}

gint GRBF_FUNCTION_NAME(grbf_grid_count_points_2d)(GRBF_REAL *F, gint fstr,
						   gint ni, gint ldf,
						   gint nj, gint nc,
						   GRBF_REAL tol)

{
  gint i, j, k, n ;
  GRBF_REAL norm ;
  
  n = 0 ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      norm = 0 ;
      for ( k = 0 ; k < nc ; k ++ ) {
	norm += F[i*ldf + j*fstr + k]*F[i*ldf + j*fstr + k] ;
      }
      if ( norm > tol*tol ) n ++ ;
    }
  }
  
  return n ;
}

gint GRBF_FUNCTION_NAME(grbf_grid_increment_3d)(GRBF_REAL *F, gint fstr,
						gint ni, gint ldfi,
						gint nj, gint ldfj,
						gint nk, gint nc,
						GRBF_REAL *origin, GRBF_REAL h,
						GRBF_REAL *x, GRBF_REAL s,
						GRBF_REAL *q, GRBF_REAL tol)

{
  gint i, j, k, i0, j0, k0, m, idx, di ;
  GRBF_REAL qmax, len, y[3], r2, E ;
  
  i0 = (gint)floor((x[0] - origin[0])/h) ;
  j0 = (gint)floor((x[1] - origin[1])/h) ;
  k0 = (gint)floor((x[2] - origin[2])/h) ;

  if ( i0 < 0 || i0 >= ni ) return 0 ;
  if ( j0 < 0 || j0 >= nj ) return 0 ;
  if ( k0 < 0 || k0 >= nk ) return 0 ;
  
  /* if ( i0 < 0 || i0 >= ni ) { */
  /*   g_error("%s: x coordinate (%lg) out of range (%lg--%lg), i = %d", */
  /* 	    __FUNCTION__, x[0], origin[0], origin[0] + h*ni, i0) ; */
  /* } */
  /* if ( j0 < 0 || j0 >= nj ) { */
  /*   g_error("%s: y coordinate (%lg) out of range (%lg--%lg), j = %d", */
  /* 	    __FUNCTION__, x[1], origin[1], origin[1] + h*nj, j0) ; */
  /* } */
  /* if ( k0 < 0 || k0 >= nk ) { */
  /*   g_error("%s: z coordinate (%lg) out of range (%lg--%lg), k = %d", */
  /* 	    __FUNCTION__, x[2], origin[2], origin[2] + h*nk, k0) ; */
  /* } */

  qmax = 0 ;
  for ( m = 0 ; m < nc ; m ++ ) { qmax = MAX(qmax, fabs(q[m])) ; }

  if ( qmax < tol ) return 0 ;

  /*qmax*exp(-len*len/s/s) = tol*/
  len = sqrt(-s*s*log(tol/qmax)) ;
  di = (gint)ceil(len/h) ;

  for ( i = MAX(0,i0-di) ; i < MIN(ni,i0+di) ; i ++ ) {
    y[0] = origin[0] + h*i ;
    for ( j = MAX(0,j0-di) ; j < MIN(nj,j0+di) ; j ++ ) {
      y[1] = origin[1] + h*j ;
      for ( k = MAX(0,k0-di) ; k < MIN(nk,k0+di) ; k ++ ) {
	y[2] = origin[2] + h*k ;
	r2 = grbf_vector3d_distance2(x,y) ;
	E = exp(-r2/s/s) ;
	idx = i*ldfi + j*ldfj + k*fstr ;
	for ( m = 0 ; m < nc ; m ++ ) {
	  F[idx + m] += q[m]*E ;
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
