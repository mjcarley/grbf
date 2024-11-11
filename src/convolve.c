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

#include <fftw3.h>

#include <grbf.h>

#include "grbf-private.h"

/**
 *
 * @ingroup weights
 * 
 * @{
 * 
 */

static inline void vector_mul_complex(gint n, GRBF_REAL al,
				      GRBF_REAL *x, gint xstr,
				      GRBF_REAL *y, gint ystr)

/*x := al*x.*y; strides are over complex elements*/

{
  gint i ;
  GRBF_REAL tmp ;

  for ( i = 0 ; i < n ; i ++ ) {
    tmp = x[2*xstr*i+0]*y[2*ystr*i+0] - x[2*xstr*i+1]*y[2*ystr*i+1] ;
    x[2*xstr*i+1] = al*(x[2*xstr*i+0]*y[2*ystr*i+1] +
			x[2*xstr*i+1]*y[2*ystr*i+0]) ; 
    x[2*xstr*i+0] = al*tmp ;
  }
  
  return ;
}

/** 
 * Evaluate weights of a Gaussian Radial Basis Function fit in one
 * dimension.
 * 
 * @param F function to fit;
 * @param fstr stride between entries in \a F;
 * @param nf number of entries in \a F;
 * @param nc number of components per entry;
 * @param E coefficients of cardinal function for Gaussian fit;
 * @param N number of elements in \a E is \c N+1 or \c (2N+1), depending 
 * on \a duplicated;
 * @param duplicated if TRUE, \a E contains entries from -N to N, otherwise,
 * from 0 to N;
 * @param w on output contains weights of fit;
 * @param wstr stride of entries in \a w.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_interpolation_weights_1d_fft)(GRBF_REAL *F,
							   gint fstr, gint nf,
							   gint nc,
							   GRBF_REAL *E,
							   gint N,
							   gboolean duplicated,
							   GRBF_REAL *w,
							   gint wstr)

{
  gint i, j ;
#ifdef GRBF_SINGLE_PRECISION
  fftwf_plan fp, ip ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_plan fp, ip ;
#endif /*GRBF_SINGLE_PRECISION*/
  gint rank, np[1], howmany, *inembed, istride, idist,
    *onembed, ostride, odist ;
  guint flags ;
  
  g_assert(fstr >= nc) ;
  g_assert(wstr >= nc) ;

  /*Fourier convolution*/
  memset(w, 0, nf*sizeof(GRBF_REAL)) ;
  if ( duplicated ) {
    for ( j = 0 ; j < nc ; j ++ ) {
      w[0*wstr+j] = E[N] ;
      for ( i = 1 ; i <= N ; i ++ ) {
	w[i*wstr+j] = w[(nf-i)*wstr+j] = E[N+i] ;
      }
    }
  } else {
    for ( j = 0 ; j < nc ; j ++ ) {
      w[0*wstr+j] = E[0] ;
      for ( i = 1 ; i <= N ; i ++ ) {
	w[i*wstr+j] = w[(nf-i)*wstr+j] = E[i] ;
      }
    }
  }

  /*http://fftw.org/fftw3_doc/Advanced-Real_002ddata-DFTs.html*/
  rank = 1 ; np[0] = nf ; howmany = nc ;
  inembed = NULL ; istride = fstr ; idist = 1 ;
  onembed = NULL ; ostride = fstr ; odist = 1 ;
  flags = FFTW_ESTIMATE ;
#ifdef GRBF_SINGLE_PRECISION
  fp = fftwf_plan_many_dft_r2c(rank, np, howmany,
			      F, inembed, istride, idist,
			      (fftwf_complex *)F, onembed, ostride, odist,
			      flags) ;
  ip = fftwf_plan_many_dft_c2r(rank, np, howmany,
			      (fftwf_complex *)F, inembed, istride, idist,
			      F, onembed, ostride, odist,
			      flags) ;

  fftwf_execute_dft_r2c(fp, F, (fftwf_complex *)F) ;
  fftwf_execute_dft_r2c(fp, w, (fftwf_complex *)w) ;  
#else /*GRBF_SINGLE_PRECISION*/
  fp = fftw_plan_many_dft_r2c(rank, np, howmany,
			      F, inembed, istride, idist,
			      (fftw_complex *)F, onembed, ostride, odist,
			      flags) ;
  ip = fftw_plan_many_dft_c2r(rank, np, howmany,
			      (fftw_complex *)F, inembed, istride, idist,
			      F, onembed, ostride, odist,
			      flags) ;

  fftw_execute_dft_r2c(fp, F, (fftw_complex *)F) ;
  fftw_execute_dft_r2c(fp, w, (fftw_complex *)w) ;
#endif /*GRBF_SINGLE_PRECISION*/

  for ( j = 0 ; j < nc ; j ++ ) {
    vector_mul_complex(nf/2+1, 1.0/nf, &(w[2*j]), wstr, &(F[2*j]), fstr) ;
  }
  
#ifdef GRBF_SINGLE_PRECISION
  fftwf_execute_dft_c2r(ip, (fftwf_complex *)w, w) ;
  fftwf_execute_dft_c2r(ip, (fftwf_complex *)F, F) ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_execute_dft_c2r(ip, (fftw_complex *)w, w) ;
  fftw_execute_dft_c2r(ip, (fftw_complex *)F, F) ;
#endif /*GRBF_SINGLE_PRECISION*/
  
  return 0 ;
}

/** 
 * Evaluate weights of a Gaussian Radial Basis Function fit in one
 * dimension, using FFT.
 * 
 * @param F function to fit;
 * @param fstr stride between entries in \a F;
 * @param nf number of entries in \a F;
 * @param nc number of components per entry;
 * @param w workspace allocated with ::grbf_workspace_alloc and 
 * initialised with ::grbf_workspace_init_1d;
 * @param wt on output contains weights of expansion;
 * @param wtstr stride of data in \a wt.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_interpolation_weights_fft_1d)(GRBF_REAL *F,
							   gint fstr, gint nf,
							   gint nc,
							   grbf_workspace_t *w,
							   GRBF_REAL *wt,
							   gint wtstr)

{
  gint j ;
  GRBF_REAL *Ef ;

  if ( grbf_workspace_dimension(w) != 1 ) {
    g_error("%s: workplace dimension (%d) not equal to one",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    

  if ( grbf_workspace_data_size(w) != sizeof(GRBF_REAL) ) {
    g_error("%s: workspace assigned for wrong floating point type",
	    __FUNCTION__) ;
  }

#ifdef GRBF_SINGLE_PRECISION
  fftwf_execute_dft_r2c(grbf_workspace_plan_forward_f(w,0),
			F, (fftwf_complex *)wt) ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,0),
		       F, (fftw_complex *)wt) ;
#endif /*GRBF_SINGLE_PRECISION*/
  
  Ef = (GRBF_REAL *)(w->Ef) ;
  for ( j = 0 ; j < nc ; j ++ ) {
    vector_mul_complex(nf/2+1, 1.0, &(wt[2*j]), wtstr, Ef, 1) ;
  }

#ifdef GRBF_SINGLE_PRECISION
  fftwf_execute_dft_c2r(grbf_workspace_plan_inverse_f(w,0),
			(fftwf_complex *)wt, wt) ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,0),
		       (fftw_complex *)wt, wt) ;
#endif /*GRBF_SINGLE_PRECISION*/
  
  return 0 ;
}

static void cardinal_coefficients_fft(gint N, GRBF_REAL al,
				      GRBF_REAL *E, gint ne,
				      GRBF_REAL *work)

{
  gint i ;
#ifdef GRBF_SINGLE_PRECISION
  fftwf_plan p ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_plan p ;
#endif /*GRBF_SINGLE_PRECISION*/
  memset(E, 0, ne*sizeof(GRBF_REAL)) ;
  GRBF_FUNCTION_NAME(grbf_cardinal_function_coefficients)(al, N, E, FALSE,
							  NULL, work) ;
  E[0] /= ne ;
  for ( i = 1 ; i <= N ; i ++ ) {
   E[   i] /= ne ;
   E[ne-i] = E[i] ;
  }

#ifdef GRBF_SINGLE_PRECISION
  p = fftwf_plan_dft_r2c_1d(ne, E, (fftwf_complex *)E, FFTW_ESTIMATE) ;
  fftwf_execute(p) ;
  fftwf_destroy_plan(p) ;
#else /*GRBF_SINGLE_PRECISION*/
  p = fftw_plan_dft_r2c_1d(ne, E, (fftw_complex *)E, FFTW_ESTIMATE) ;
  fftw_execute(p) ;
  fftw_destroy_plan(p) ;
#endif /*GRBF_SINGLE_PRECISION*/
  
  return ;
}

/** 
 * Initialise a workspace for evaluation of Gaussian RBF weights on a
 * regular grid. The \f$j\f$th element of entry \c i in \a F is
 * indexed as \c F[i*fstr+j], and similarly for \a wt.
 *
 * @param w workspace to be initialised;
 * @param F input function array;
 * @param fstr data stride in \a F;
 * @param nf number of points in \a F;
 * @param wt Gaussian weight array
 * @param wtstr data stride in \a wt;
 * @param nc number of components per entry in \a F;
 * @param al overlap ratio \f$\alpha\f$;
 * @param N number of cardinal function points (width of convolution 
 * window); 
 * @param work workspace for evaluation of cardinal function, with at
 * least \f$(N+1)^2+3N+1\f$ elements.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_workspace_init_1d)(grbf_workspace_t *w,
						GRBF_REAL *F, gint fstr,
						gint nf,
						GRBF_REAL *wt, gint wtstr,
						gint nc, GRBF_REAL al, gint N,
						GRBF_REAL *work)

{
  guint flags ;
  gint rank, np[1], howmany, *inembed, istride, idist,
    *onembed, ostride, odist ;
  
  if ( grbf_workspace_dimension(w) != 1 ) {
    g_error("%s: workplace dimension (%d) not equal to one",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    
  if ( nf > grbf_workspace_length(w) ) {
    g_error("%s: not enough space (%d) for %d points",
	    __FUNCTION__, grbf_workspace_length(w), nf) ;
  }
  if ( 2*N > nf ) {
    g_error("%s: number of weights (%d) must be less than "
	    "half length of F (%d)", __FUNCTION__, N, nf) ;
  }  
  if ( grbf_workspace_data_size(w) != sizeof(GRBF_REAL) ) {
    g_error("%s: workspace assigned for wrong floating point type",
	    __FUNCTION__) ;
  }

  /*generate Fourier transform of convolution weights*/
  cardinal_coefficients_fft(N, al, (GRBF_REAL *)(w->Ef), nf, work) ;

  rank = 1 ; np[0] = nf ; howmany = nc ;
  inembed = NULL ; istride = fstr ; idist = 1 ;
  onembed = NULL ; ostride = wtstr ; odist = 1 ;
  flags = FFTW_ESTIMATE ;
#ifdef GRBF_SINGLE_PRECISION
  w->fpf[0] = fftwf_plan_many_dft_r2c(rank, np, howmany,
				      F, inembed, istride, idist,
				      (fftwf_complex *)wt, onembed,
				      ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->fp[0] = fftw_plan_many_dft_r2c(rank, np, howmany,
				    F, inembed, istride, idist,
				    (fftw_complex *)wt, onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/
  
  inembed = NULL ; istride = wtstr ; idist = 1 ;
  onembed = NULL ; ostride = wtstr ; odist = 1 ;
#ifdef GRBF_SINGLE_PRECISION
  w->ipf[0] = fftwf_plan_many_dft_c2r(rank, np, howmany,
				      (fftwf_complex *)wt, inembed,
				      istride, idist,
				      wt, onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->ip[0] = fftw_plan_many_dft_c2r(rank, np, howmany,
				    (fftw_complex *)wt, inembed, istride, idist,
				    wt, onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/
  
  return 0 ;
}

/** 
 * Evaluate weights of a Gaussian Radial Basis Function fit in two
 * dimensions, using FFT. The \f$k\f$th element of entry \c (i,j) in
 * \a F is indexed as \c F[i*ldf+j*fstr+k], and similarly for \a wt.
 * 
 * @param F function to fit;
 * @param fstr stride between entries in \a F;
 * @param ni number of entries in first dimension of \a F;
 * @param ldf leading dimension of \a F;
 * @param nj number of entries in second dimension of \a F;
 * @param nc number of components per entry;
 * @param w workspace allocated with ::grbf_workspace_alloc and 
 * initialised with ::grbf_workspace_init_2d;
 * @param wt on output contains weights of expansion;
 * @param wtstr stride of data in \a wt,
 * @param ldw leading dimension of \a wt.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_interpolation_weights_fft_2d)(GRBF_REAL *F,
							   gint fstr,
							   gint ni, gint ldf,
							   gint nj, gint nc,
							   grbf_workspace_t *w,
							   GRBF_REAL *wt,
							   gint wtstr, gint ldw)

{
  gint i, j, k ;
  GRBF_REAL *Ef, *buf ;
#ifdef GRBF_SINGLE_PRECISION
  fftwf_complex *cbuf ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_complex *cbuf ;
#endif /*GRBF_SINGLE_PRECISION*/
  
  if ( grbf_workspace_dimension(w) != 2 ) {
    g_error("%s: workplace dimension (%d) not equal to two",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    

  if ( grbf_workspace_data_size(w) != sizeof(GRBF_REAL) ) {
    g_error("%s: workspace assigned for wrong floating point type",
	    __FUNCTION__) ;
  }
  
  Ef = w->Ef ; buf = w->buf ; cbuf = w->buf ;
  for ( j = 0 ; j < nj ; j ++ ) {
    for ( k = 0 ; k < nc ; k ++ ) {
#ifdef GRBF_SINGLE_PRECISION
      fftwf_execute_dft_r2c(grbf_workspace_plan_forward_f(w,0),
			    &(F[j*fstr+k]), cbuf) ;
      vector_mul_complex(ni/2+1, 1.0, buf, 1, Ef, 1) ;
      
      fftwf_execute_dft_c2r(grbf_workspace_plan_inverse_f(w,0),
			    cbuf, &(wt[j*wtstr+k])) ;
#else /*GRBF_SINGLE_PRECISION*/
      fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,0),
			   &(F[j*fstr+k]), cbuf) ;
      vector_mul_complex(ni/2+1, 1.0, buf, 1, Ef, 1) ;
      
      fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,0),
			   cbuf, &(wt[j*wtstr+k])) ;
#endif /*GRBF_SINGLE_PRECISION*/
    }
  }
  
#if 0
  /*test code for checking FFTs*/
  {GRBF_REAL err = 0 ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      err = MAX(err, fabs(wt[i*ldw+j*wtstr]/ni - F[i*ldf+j*fstr])) ;
      /* err = MAX(err, fabs(wt[i*ldw+j]/ni - F[i*ldf+j])) ; */
      /* err = MAX(err, fabs(wt[i*ldw+j] - F[i*ldf+j])) ; */
    }
  }
  fprintf(stderr, "error: %lg\n", err) ;
  }
#endif

  Ef = &(Ef[ni+2]) ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( k = 0 ; k < nc ; k ++ ) {
#ifdef GRBF_SINGLE_PRECISION
      fftwf_execute_dft_r2c(grbf_workspace_plan_forward_f(w,1),
			    &(wt[i*ldw+k]), cbuf) ;
      vector_mul_complex(nj/2+1, 1.0, buf, 1, Ef, 1) ;
      
      fftwf_execute_dft_c2r(grbf_workspace_plan_inverse_f(w,1),
			    cbuf, &(wt[i*ldw+k])) ;
#else /*GRBF_SINGLE_PRECISION*/
      fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,1),
			   &(wt[i*ldw+k]), cbuf) ;
      vector_mul_complex(nj/2+1, 1.0, buf, 1, Ef, 1) ;
      
      fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,1),
			   cbuf, &(wt[i*ldw+k])) ;
#endif /*GRBF_SINGLE_PRECISION*/
    }
  }

#if 0
  /*test code for checking FFTs*/
  {
  GRBF_REAL err = 0 ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      err = MAX(err, fabs(wt[i*ldw+j*wtstr]/ni/nj - F[i*ldf+j*fstr])) ;
      /* err = MAX(err, fabs(wt[i*ldw+j] - F[i*ldf+j])) ; */
    }
  }
  fprintf(stderr, "error: %lg\n", err) ;
  }
#endif
  
  return 0 ;
}

/** 
 * Initialise a workspace for evaluation of Gaussian RBF weights on a
 * regular two-dimensional grid. The \f$k\f$th element of entry \c
 * (i,j) in \a F is indexed as \c F[i*ldf+j*fstr+k], and similarly for \a wt. 
 *
 * @param w workspace to be initialised;
 * @param F input function array;
 * @param fstr data stride in \a F;
 * @param ni number of points in \f$i\f$ grid direction;
 * @param ldf leading dimension of \a F;
 * @param nj number of points in \f$j\f$ grid direction;
 * @param wt Gaussian weight array
 * @param wtstr data stride in \a wt;
 * @param ldw leading dimension of \a wt;
 * @param al overlap ratio \f$\alpha\f$;
 * @param N number of cardinal function points (width of convolution 
 * window); 
 * @param work workspace for evaluation of cardinal function, with at
 * least \f$(N+1)^2+3N+1\f$ elements.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_workspace_init_2d)(grbf_workspace_t *w,
						GRBF_REAL *F, gint fstr,
						gint ni, gint ldf, gint nj,
						GRBF_REAL *wt, gint wtstr,
						gint ldw,
						GRBF_REAL al, gint N,
						GRBF_REAL *work)

{
  guint flags ;
  gint rank, howmany, *inembed, istride, idist,
    *onembed, ostride, odist ;
#ifdef GRBF_SINGLE_PRECISION
  fftwf_complex *cbuf ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_complex *cbuf ;
#endif /*GRBF_SINGLE_PRECISION*/
  GRBF_REAL *Ef ;
  
  if ( grbf_workspace_dimension(w) != 2 ) {
    g_error("%s: workplace dimension (%d) not equal to two",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    
  if ( ni > grbf_workspace_length(w) ) {
    g_error("%s: not enough space (%d) for %d points in i",
	    __FUNCTION__, grbf_workspace_length(w), ni) ;
  }
  if ( nj > grbf_workspace_length(w) ) {
    g_error("%s: not enough space (%d) for %d points in j",
	    __FUNCTION__, grbf_workspace_length(w), nj) ;
  }
  if ( 2*N > ni || 2*N > nj ) {
    g_error("%s: number of weights (%d) must be less than "
	    "half minimum dimension (%d)", __FUNCTION__, N, MIN(ni,nj)) ;
  }
  if ( grbf_workspace_data_size(w) != sizeof(GRBF_REAL) ) {
    g_error("%s: workspace assigned for wrong floating point type",
	    __FUNCTION__) ;
  }

  /*generate Fourier transform of convolution weights*/
  Ef = w->Ef ;
  cardinal_coefficients_fft(N, al, &(Ef[   0]), ni, work) ;
  cardinal_coefficients_fft(N, al, &(Ef[ni+2]), nj, work) ;

#ifdef GRBF_SINGLE_PRECISION
  cbuf = (fftwf_complex *)(w->buf) ;
#else /*GRBF_SINGLE_PRECISION*/
  cbuf = (fftw_complex *)(w->buf) ;
#endif /*GRBF_SINGLE_PRECISION*/

  /*FFTs for convolution in i: F[i*ldf+j*fstr+k]*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = ldf ; idist = fstr ;
  onembed = NULL ; ostride = 1   ; odist = 1    ;
  flags = FFTW_ESTIMATE ;
#ifdef GRBF_SINGLE_PRECISION
  w->fpf[0] = fftwf_plan_many_dft_r2c(rank, &ni, howmany,
				      F   , inembed, istride, idist,
				      cbuf, onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->fp[0] = fftw_plan_many_dft_r2c(rank, &ni, howmany,
				    F   , inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  inembed = NULL ; istride = 1   ; idist = 1 ;
  onembed = NULL ; ostride = ldw ; odist = wtstr ;
#ifdef GRBF_SINGLE_PRECISION
  w->ipf[0] = fftwf_plan_many_dft_c2r(rank, &ni, howmany,
				      cbuf, inembed, istride, idist,
				      wt  , onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->ip[0] = fftw_plan_many_dft_c2r(rank, &ni, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/
  /*FFTs for convolution in j*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = wtstr ; idist = ldw ;
  onembed = NULL ; ostride = 1     ; odist = 1   ;
  flags = FFTW_ESTIMATE ;
#ifdef GRBF_SINGLE_PRECISION
  w->fpf[1] = fftwf_plan_many_dft_r2c(rank, &nj, howmany,
				      wt, inembed, istride, idist,
				      cbuf, onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->fp[1] = fftw_plan_many_dft_r2c(rank, &nj, howmany,
				    wt, inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  inembed = NULL ; istride = 1     ; idist = 1   ;
  onembed = NULL ; ostride = wtstr ; odist = ldw ;
#ifdef GRBF_SINGLE_PRECISION
  w->ipf[1] = fftwf_plan_many_dft_c2r(rank, &nj, howmany,
				      cbuf, inembed, istride, idist,
				      wt  , onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->ip[1] = fftw_plan_many_dft_c2r(rank, &nj, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  return 0 ;
}

/** 
 * Initialise a workspace for evaluation of Gaussian RBF weights on a
 * regular three-dimensional grid. The \f$m\f$th element of entry \c
 * (i,j,k) in \a F is indexed as \c F[i*ldfi+j*ldfj+k*fstr+m], and
 * similarly for \a wt.
 *
 * @param w workspace to be initialised;
 * @param F input function array;
 * @param fstr data stride in \a F;
 * @param ni number of points in \f$i\f$ grid direction;
 * @param ldfi leading dimension of \a F in \c i;
 * @param nj number of points in \f$j\f$ grid direction;
 * @param ldfj leading dimension of \a F in \c j;
 * @param nk number of points in \f$k\f$ grid direction;
 * @param wt Gaussian weight array
 * @param wtstr data stride in \a wt;
 * @param ldwi leading dimension of \a wt in \c i;
 * @param ldwj leading dimension of \a wt in \c j;
 * @param al overlap ratio \f$\alpha\f$;
 * @param N number of cardinal function points (width of convolution 
 * window); 
 * @param work workspace for evaluation of cardinal function, with at
 * least \f$(N+1)^2+3N+1\f$ elements.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_workspace_init_3d)(grbf_workspace_t *w,
						GRBF_REAL *F, gint fstr,
						gint ni, gint ldfi,
						gint nj, gint ldfj,
						gint nk,
						GRBF_REAL *wt, gint wtstr,
						gint ldwi, gint ldwj,
						GRBF_REAL al, gint N,
						GRBF_REAL *work)

{
  guint flags ;
  gint rank, howmany, *inembed, istride, idist,
    *onembed, ostride, odist ;
#ifdef GRBF_SINGLE_PRECISION
  fftwf_complex *cbuf ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_complex *cbuf ;
#endif /*GRBF_SINGLE_PRECISION*/
  GRBF_REAL *Ef ;

  if ( grbf_workspace_dimension(w) != 3 ) {
    g_error("%s: workplace dimension (%d) not equal to three",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    
  if ( ni > grbf_workspace_length(w) ) {
    g_error("%s: not enough space (%d) for %d points in i",
	    __FUNCTION__, grbf_workspace_length(w), ni) ;
  }
  if ( nj > grbf_workspace_length(w) ) {
    g_error("%s: not enough space (%d) for %d points in j",
	    __FUNCTION__, grbf_workspace_length(w), nj) ;
  }
  if ( nk > grbf_workspace_length(w) ) {
    g_error("%s: not enough space (%d) for %d points in k",
	    __FUNCTION__, grbf_workspace_length(w), nk) ;
  }
  if ( 2*N > ni || 2*N > nj || 2*N > nk ) {
    g_error("%s: number of weights (%d) must be less than "
	    "half minimum dimension (%d)",
	    __FUNCTION__, N, MIN(ni,MIN(nj,nk))) ;
  }

  if ( grbf_workspace_data_size(w) != sizeof(GRBF_REAL) ) {
    g_error("%s: workspace assigned for wrong floating point type",
	    __FUNCTION__) ;
  }

  /*generate Fourier transform of convolution weights*/
  Ef = w->Ef ;
  cardinal_coefficients_fft(N, al, &(Ef[      0]), ni, work) ;
  cardinal_coefficients_fft(N, al, &(Ef[   ni+2]), nj, work) ;
  cardinal_coefficients_fft(N, al, &(Ef[ni+nj+4]), nk, work) ;

  cbuf = w->buf ;
  /*FFTs for convolution in i: F[i*ldfi+j*ldfj+k*fstr]*/

  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = ldfi ; idist = fstr ;
  onembed = NULL ; ostride = 1    ; odist = 1    ;
  flags = FFTW_ESTIMATE ;
#ifdef GRBF_SINGLE_PRECISION
  w->fpf[0] = fftwf_plan_many_dft_r2c(rank, &ni, howmany,
				      F   , inembed, istride, idist,
				      cbuf, onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->fp[0] = fftw_plan_many_dft_r2c(rank, &ni, howmany,
				    F   , inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  inembed = NULL ; istride = 1    ; idist = 1 ;
  onembed = NULL ; ostride = ldwi ; odist = wtstr ;
#ifdef GRBF_SINGLE_PRECISION
  w->ipf[0] = fftwf_plan_many_dft_c2r(rank, &ni, howmany,
				      cbuf, inembed, istride, idist,
				      wt  , onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->ip[0] = fftw_plan_many_dft_c2r(rank, &ni, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  /*FFTs for convolution in j*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = ldwj ; idist = 1 ;
  onembed = NULL ; ostride = 1    ; odist = 1 ;
  flags = FFTW_ESTIMATE ;
#ifdef GRBF_SINGLE_PRECISION
  w->fpf[1] = fftwf_plan_many_dft_r2c(rank, &nj, howmany,
				      wt  , inembed, istride, idist,
				      cbuf, onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->fp[1] = fftw_plan_many_dft_r2c(rank, &nj, howmany,
				    wt  , inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  inembed = NULL ; istride = 1    ; idist = 1 ;
  onembed = NULL ; ostride = ldwj ; odist = wtstr ;
#ifdef GRBF_SINGLE_PRECISION
  w->ipf[1] = fftwf_plan_many_dft_c2r(rank, &nj, howmany,
				      cbuf, inembed, istride, idist,
				      wt  , onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->ip[1] = fftw_plan_many_dft_c2r(rank, &nj, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  /*FFTs for convolution in k*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = wtstr ; idist = 1 ;
  onembed = NULL ; ostride = 1     ; odist = 1 ;
  flags = FFTW_ESTIMATE ;
#ifdef GRBF_SINGLE_PRECISION
  w->fpf[2] = fftwf_plan_many_dft_r2c(rank, &nk, howmany,
				      wt, inembed, istride, idist,
				      cbuf, onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->fp[2] = fftw_plan_many_dft_r2c(rank, &nk, howmany,
				    wt, inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/
  inembed = NULL ; istride = 1     ; idist = 1 ;
  onembed = NULL ; ostride = wtstr ; odist = 1 ;
#ifdef GRBF_SINGLE_PRECISION
  w->ipf[2] = fftwf_plan_many_dft_c2r(rank, &nk, howmany,
				      cbuf, inembed, istride, idist,
				      wt  , onembed, ostride, odist,
				      flags) ;
#else /*GRBF_SINGLE_PRECISION*/
  w->ip[2] = fftw_plan_many_dft_c2r(rank, &nk, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
#endif /*GRBF_SINGLE_PRECISION*/

  return 0 ;
}

/** 
 * Evaluate weights of a Gaussian Radial Basis Function fit in three
 * dimensions, using FFT. The \f$m\f$th element of entry \c (i,j,k) in
 * \a F is indexed as \c F[i*ldfi+j*ldfj+k*fstr+m], and similarly for
 * \a wt.
 * 
 * @param F function to fit;
 * @param fstr stride between entries in \a F;
 * @param ni number of entries in first dimension of \a F;
 * @param ldfi leading dimension of \a F;
 * @param nj number of entries in second dimension of \a F;
 * @param ldfj second dimension of \a F;
 * @param nk number of entries in third dimension of \a F;
 * @param nc number of components per entry;
 * @param w workspace allocated with ::grbf_workspace_alloc and 
 * initialised with ::grbf_workspace_init_2d;
 * @param wt on output contains weights of expansion;
 * @param wtstr stride of data in \a wt,
 * @param ldwi leading dimension of \a wt;
 * @param ldwj second dimension of \a wt.
 * 
 * @return 0 on success.
 */

gint GRBF_FUNCTION_NAME(grbf_interpolation_weights_fft_3d)(GRBF_REAL *F,
							   gint fstr,
							   gint ni, gint ldfi,
							   gint nj, gint ldfj,
							   gint nk, gint nc,
							   grbf_workspace_t *w,
							   GRBF_REAL *wt,
							   gint wtstr,
							   gint ldwi, gint ldwj)
{
  gint i, j, k, m ;
  GRBF_REAL *Ef, *buf ;
#ifdef GRBF_SINGLE_PRECISION
  fftwf_complex *cbuf ;
#else /*GRBF_SINGLE_PRECISION*/
  fftw_complex *cbuf ;
#endif /*GRBF_SINGLE_PRECISION*/

  if ( grbf_workspace_dimension(w) != 3 ) {
    g_error("%s: workplace dimension (%d) not equal to three",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    
  if ( grbf_workspace_data_size(w) != sizeof(GRBF_REAL) ) {
    g_error("%s: workspace assigned for wrong floating point type",
	    __FUNCTION__) ;
  }

  Ef = w->Ef ; buf = w->buf ; cbuf = w->buf ;
  for ( j = 0 ; j < nj ; j ++ ) {
    for ( k = 0 ; k < nk ; k ++ ) {
      for ( m = 0 ; m < nc ; m ++ ) {
  
#ifdef GRBF_SINGLE_PRECISION
	fftwf_execute_dft_r2c(grbf_workspace_plan_forward_f(w,0),
			     &(F[j*ldfj+k*fstr+m]), cbuf) ;
	vector_mul_complex(ni/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftwf_execute_dft_c2r(grbf_workspace_plan_inverse_f(w,0),
			     cbuf, &(wt[j*ldwj+k*wtstr+m])) ;
#else /*GRBF_SINGLE_PRECISION*/
	fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,0),
			     &(F[j*ldfj+k*fstr+m]), cbuf) ;
	vector_mul_complex(ni/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,0),
			     cbuf, &(wt[j*ldwj+k*wtstr+m])) ;
#endif /*GRBF_SINGLE_PRECISION*/
      }
    }
  }
  
#if 0
  /*test code for checking FFTs*/
  {GRBF_REAL err = 0 ;
    for ( i = 0 ; i < ni ; i ++ ) {
      for ( j = 0 ; j < nj ; j ++ ) {
	for ( k = 0 ; k < nk ; k ++ ) {
	  err = MAX(err, fabs(wt[i*ldwi+j*ldwj+k*wtstr]/ni -
			      F[i*ldfi+j*ldfj+k*fstr])) ;
	}
      }
    }
    fprintf(stderr, "error: %lg\n", err) ;
  }
#endif

  Ef = &(Ef[ni+2]) ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( k = 0 ; k < nk ; k ++ ) {
      for ( m = 0 ; m < nc ; m ++ ) {
#ifdef GRBF_SINGLE_PRECISION
	fftwf_execute_dft_r2c(grbf_workspace_plan_forward_f(w,1),
			      &(wt[i*ldwi+k*wtstr+m]), cbuf) ;
	vector_mul_complex(nj/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftwf_execute_dft_c2r(grbf_workspace_plan_inverse_f(w,1),
			      cbuf, &(wt[i*ldwi+k*wtstr+m])) ;
#else /*GRBF_SINGLE_PRECISION*/
	fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,1),
			     &(wt[i*ldwi+k*wtstr+m]), cbuf) ;
	vector_mul_complex(nj/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,1),
			     cbuf, &(wt[i*ldwi+k*wtstr+m])) ;
#endif /*GRBF_SINGLE_PRECISION*/
      }
    }
  }

#if 0
  /*test code for checking FFTs*/
  {GRBF_REAL err = 0 ;
    for ( i = 0 ; i < ni ; i ++ ) {
      for ( j = 0 ; j < nj ; j ++ ) {
	for ( k = 0 ; k < nk ; k ++ ) {
	  err = MAX(err, fabs(wt[i*ldwi+j*ldwj+k*wtstr]/ni/nj -
			      F[i*ldfi+j*ldfj+k*fstr])) ;
	}
      }
    }
    fprintf(stderr, "error: %lg\n", err) ;
  }
#endif

  Ef = &(Ef[nj+2]) ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      for ( m = 0 ; m < nc ; m ++ ) {
#ifdef GRBF_SINGLE_PRECISION
	fftwf_execute_dft_r2c(grbf_workspace_plan_forward_f(w,2),
			      &(wt[i*ldwi+j*ldwj+m]), cbuf) ;
	vector_mul_complex(nk/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftwf_execute_dft_c2r(grbf_workspace_plan_inverse_f(w,2),
			      cbuf, &(wt[i*ldwi+j*ldwj+m])) ;
#else /*GRBF_SINGLE_PRECISION*/
	fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,2),
			     &(wt[i*ldwi+j*ldwj+m]), cbuf) ;
	vector_mul_complex(nk/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,2),
			     cbuf, &(wt[i*ldwi+j*ldwj+m])) ;
#endif /*GRBF_SINGLE_PRECISION*/
      }
    }
  }

#if 0
  /*test code for checking FFTs*/
  {GRBF_REAL err = 0 ;
    for ( i = 0 ; i < ni ; i ++ ) {
      for ( j = 0 ; j < nj ; j ++ ) {
	for ( k = 0 ; k < nk ; k ++ ) {
	  err = MAX(err, fabs(wt[i*ldwi+j*ldwj+k*wtstr]/ni/nj/nk -
			      F[i*ldfi+j*ldfj+k*fstr])) ;
	}
      }
    }
    fprintf(stderr, "error: %lg\n", err) ;
  }
#endif
  
  return 0 ;
}

/** 
 * Allocate a new workspace for evaluation of interpolation
 * weights. The workspace needs to be initialised with
 * ::grbf_workspace_init_1d, ::grbf_workspace_init_2d, or
 * ::grbf_workspace_init_3d, before being used to find weights.
 * 
 * @param dim dimension of problem, 1, 2, or 3;
 * @param len maximum number of points per dimension;
 * 
 * @return newly allocated ::grbf_workspace_t.
 */

grbf_workspace_t *GRBF_FUNCTION_NAME(grbf_workspace_alloc)(gint dim, gint len)

{
  grbf_workspace_t *w ;
  GRBF_REAL *buf ;
  
  w = (grbf_workspace_t *)g_malloc0(sizeof(grbf_workspace_t)) ;
  
  grbf_workspace_dimension(w) = dim ;
  grbf_workspace_data_size(w) = sizeof(GRBF_REAL) ;
  
  if ( dim == 1 ) {
    grbf_workspace_length(w)    = len ;
    buf = (GRBF_REAL *)g_malloc(3*(len+2)*sizeof(GRBF_REAL)) ;
    w->buf = buf ;
    w->Ef = &(buf[(len+2)]) ;
    return w ;
  }

  if ( dim == 2 ) {
    grbf_workspace_length(w) = len ;
    buf = (GRBF_REAL *)g_malloc(3*(len+2)*sizeof(GRBF_REAL)) ;
    w->buf = buf ;
    w->Ef = &(buf[(len+2)]) ;

    return w ;
  }

  if ( dim == 3 ) {
    grbf_workspace_length(w) = len ;
    buf = (GRBF_REAL *)g_malloc(4*(len+2)*sizeof(GRBF_REAL)) ;
    w->buf = buf ;
    w->Ef = &(buf[(len+2)]) ;

    return w ;
  }

  g_assert_not_reached() ;
  
  return w ;
}


/**
 *
 * @}
 *
 */
