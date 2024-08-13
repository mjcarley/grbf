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

static inline void vector_mul_complex(gint n, gdouble al,
				      gdouble *x, gint xstr,
				      gdouble *y, gint ystr)

/*x := al*x.*y; strides are over complex elements*/

{
  gint i ;
  gdouble tmp ;

  for ( i = 0 ; i < n ; i ++ ) {
    tmp = x[2*xstr*i+0]*y[2*ystr*i+0] - x[2*xstr*i+1]*y[2*ystr*i+1] ;
    x[2*xstr*i+1] = al*(x[2*xstr*i+0]*y[2*ystr*i+1] +
			x[2*xstr*i+1]*y[2*ystr*i+0]) ; 
    x[2*xstr*i+0] = al*tmp ;
  }
  
  return ;
}

gint grbf_interpolation_weights_1d_fft(gdouble *F, gint fstr, gint nf, gint nc,
				       gdouble *E, gint N, gboolean duplicated,
				       gdouble *w, gint wstr)

{
  gint i, j ;
  fftw_plan fp, ip ;
  gint rank, np[1], howmany, *inembed, istride, idist,
    *onembed, ostride, odist ;
  guint flags ;
  
  g_assert(fstr >= nc) ;
  g_assert(wstr >= nc) ;

  /*Fourier convolution*/
  memset(w, 0, nf*sizeof(gdouble)) ;
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

  for ( j = 0 ; j < nc ; j ++ ) {
    vector_mul_complex(nf/2+1, 1.0/nf, &(w[2*j]), wstr, &(F[2*j]), fstr) ;
  }
  
  fftw_execute_dft_c2r(ip, (fftw_complex *)w, w) ;
  fftw_execute_dft_c2r(ip, (fftw_complex *)F, F) ;
  
  return 0 ;
}

gint grbf_interpolation_weights_fft_1d(gdouble *F, gint fstr, gint nf, gint nc,
				       grbf_workspace_t *w,
				       gdouble *wt, gint wtstr)

{
  gint j ;
  gdouble *Ef ;

  if ( grbf_workspace_dimension(w) != 1 ) {
    g_error("%s: workplace dimension (%d) not equal to one",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    
  
  fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,0),
		       F, (fftw_complex *)wt) ;

  Ef = w->Ef ;
  for ( j = 0 ; j < nc ; j ++ ) {
    vector_mul_complex(nf/2+1, 1.0, &(wt[2*j]), wtstr, Ef, 1) ;
  }

  fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,0),
		       (fftw_complex *)wt, wt) ;
  
  return 0 ;
}

static void cardinal_coefficients_fft(gint N, gdouble al,
				      gdouble *E, gint ne,
				      gdouble *work)

{
  gint i ;
  fftw_plan p ;
  memset(E, 0, ne*sizeof(gdouble)) ;
  grbf_cardinal_function_coefficients(al, N, E, FALSE, NULL, work) ;
  E[0] /= ne ;
  for ( i = 1 ; i <= N ; i ++ ) {
   E[   i] /= ne ;
   E[ne-i] = E[i] ;
  }

  p = fftw_plan_dft_r2c_1d(ne, E, (fftw_complex *)E, FFTW_ESTIMATE) ;
  fftw_execute(p) ;
  fftw_destroy_plan(p) ;
  
  return ;
}

gint grbf_workspace_init_1d(grbf_workspace_t *w,
			    gdouble *F, gint fstr, gint nf,
			    gdouble *wt, gint wtstr,
			    gint nc, gdouble al, gint N, gdouble *work)

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

  /*generate Fourier transform of convolution weights*/
  cardinal_coefficients_fft(N, al, w->Ef, nf, work) ;

  rank = 1 ; np[0] = nf ; howmany = nc ;
  inembed = NULL ; istride = fstr ; idist = 1 ;
  onembed = NULL ; ostride = wtstr ; odist = 1 ;
  flags = FFTW_ESTIMATE ;
  w->fp[0] = fftw_plan_many_dft_r2c(rank, np, howmany,
				    F, inembed, istride, idist,
				    (fftw_complex *)wt, onembed, ostride, odist,
				    flags) ;
  inembed = NULL ; istride = wtstr ; idist = 1 ;
  onembed = NULL ; ostride = wtstr ; odist = 1 ;
  w->ip[0] = fftw_plan_many_dft_c2r(rank, np, howmany,
				    (fftw_complex *)wt, inembed, istride, idist,
				    wt, onembed, ostride, odist,
				    flags) ;  
  
  return 0 ;
}

gint grbf_interpolation_weights_fft_2d(gdouble *F, gint fstr,
				       gint ni, gint ldf, gint nj, gint nc,
				       grbf_workspace_t *w,
				       gdouble *wt, gint wtstr, gint ldw)

{
  gint i, j, k ;
  gdouble *Ef, *buf ;
  fftw_complex *cbuf ;
  
  if ( grbf_workspace_dimension(w) != 2 ) {
    g_error("%s: workplace dimension (%d) not equal to two",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    

  Ef = w->Ef ; buf = w->buf ; cbuf = (fftw_complex *)(w->buf) ;
  for ( j = 0 ; j < nj ; j ++ ) {
    for ( k = 0 ; k < nc ; k ++ ) {
      fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,0),
			   &(F[j*fstr+k]), cbuf) ;
      vector_mul_complex(ni/2+1, 1.0, buf, 1, Ef, 1) ;
      
      fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,0),
			   cbuf, &(wt[j*wtstr+k])) ;
    }
  }
  
#if 0
  /*test code for checking FFTs*/
  {gdouble err = 0 ;
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
      fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,1),
			   &(wt[i*ldw+k]), cbuf) ;
      vector_mul_complex(nj/2+1, 1.0, buf, 1, Ef, 1) ;
      
      fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,1),
			   cbuf, &(wt[i*ldw+k])) ;
    }
  }

#if 0
  /*test code for checking FFTs*/
  {
  gdouble err = 0 ;
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

gint grbf_workspace_init_2d(grbf_workspace_t *w,
			    gdouble *F, gint fstr, gint ni, gint ldf, gint nj,
			    gdouble *wt, gint wtstr, gint ldw,
			    gdouble al, gint N, gdouble *work)

{
  guint flags ;
  gint rank, howmany, *inembed, istride, idist,
    *onembed, ostride, odist ;
  fftw_complex *cbuf ;
  
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

  /*generate Fourier transform of convolution weights*/
  cardinal_coefficients_fft(N, al, &(w->Ef[   0]), ni, work) ;
  cardinal_coefficients_fft(N, al, &(w->Ef[ni+2]), nj, work) ;

  cbuf = (fftw_complex *)(w->buf) ;
  /*FFTs for convolution in i: F[i*ldf+j*fstr+k]*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = ldf ; idist = fstr ;
  onembed = NULL ; ostride = 1   ; odist = 1    ;
  flags = FFTW_ESTIMATE ;
  w->fp[0] = fftw_plan_many_dft_r2c(rank, &ni, howmany,
				    F   , inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
  inembed = NULL ; istride = 1   ; idist = 1 ;
  onembed = NULL ; ostride = ldw ; odist = wtstr ;
  w->ip[0] = fftw_plan_many_dft_c2r(rank, &ni, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
  /*FFTs for convolution in j*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = wtstr ; idist = ldw ;
  onembed = NULL ; ostride = 1     ; odist = 1   ;
  flags = FFTW_ESTIMATE ;
  w->fp[1] = fftw_plan_many_dft_r2c(rank, &nj, howmany,
				    wt, inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
  inembed = NULL ; istride = 1     ; idist = 1   ;
  onembed = NULL ; ostride = wtstr ; odist = ldw ;
  w->ip[1] = fftw_plan_many_dft_c2r(rank, &nj, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
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

gint grbf_workspace_init_3d(grbf_workspace_t *w,
			    gdouble *F, gint fstr,
			    gint ni, gint ldfi,
			    gint nj, gint ldfj,
			    gint nk,
			    gdouble *wt, gint wtstr,
			    gint ldwi, gint ldwj,
			    gdouble al, gint N, gdouble *work)

{
  guint flags ;
  gint rank, howmany, *inembed, istride, idist,
    *onembed, ostride, odist ;
  fftw_complex *cbuf ;
  
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

  /*generate Fourier transform of convolution weights*/
  cardinal_coefficients_fft(N, al, &(w->Ef[      0]), ni, work) ;
  cardinal_coefficients_fft(N, al, &(w->Ef[   ni+2]), nj, work) ;
  cardinal_coefficients_fft(N, al, &(w->Ef[ni+nj+4]), nk, work) ;

  cbuf = (fftw_complex *)(w->buf) ;
  /*FFTs for convolution in i: F[i*ldfi+j*ldfj+k*fstr]*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = ldfi ; idist = fstr ;
  onembed = NULL ; ostride = 1    ; odist = 1    ;
  flags = FFTW_ESTIMATE ;
  w->fp[0] = fftw_plan_many_dft_r2c(rank, &ni, howmany,
				    F   , inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
  inembed = NULL ; istride = 1    ; idist = 1 ;
  onembed = NULL ; ostride = ldwi ; odist = wtstr ;
  w->ip[0] = fftw_plan_many_dft_c2r(rank, &ni, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
  /*FFTs for convolution in j*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = ldwj ; idist = 1 ;
  onembed = NULL ; ostride = 1    ; odist = 1 ;
  flags = FFTW_ESTIMATE ;
  w->fp[1] = fftw_plan_many_dft_r2c(rank, &nj, howmany,
				    wt  , inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
  inembed = NULL ; istride = 1    ; idist = 1 ;
  onembed = NULL ; ostride = ldwj ; odist = wtstr ;
  w->ip[1] = fftw_plan_many_dft_c2r(rank, &nj, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;
  /*FFTs for convolution in k*/
  rank = 1 ; howmany = 1 ;
  inembed = NULL ; istride = wtstr ; idist = 1 ;
  onembed = NULL ; ostride = 1     ; odist = 1 ;
  flags = FFTW_ESTIMATE ;
  w->fp[2] = fftw_plan_many_dft_r2c(rank, &nk, howmany,
				    wt, inembed, istride, idist,
				    cbuf, onembed, ostride, odist,
				    flags) ;
  inembed = NULL ; istride = 1     ; idist = 1 ;
  onembed = NULL ; ostride = wtstr ; odist = 1 ;
  w->ip[2] = fftw_plan_many_dft_c2r(rank, &nk, howmany,
				    cbuf, inembed, istride, idist,
				    wt  , onembed, ostride, odist,
				    flags) ;

  return 0 ;
}

gint grbf_interpolation_weights_fft_3d(gdouble *F, gint fstr,
				       gint ni, gint ldfi,
				       gint nj, gint ldfj,
				       gint nk, gint nc,
				       grbf_workspace_t *w,
				       gdouble *wt, gint wtstr,
				       gint ldwi, gint ldwj)
{
  gint i, j, k, m ;
  gdouble *Ef, *buf ;
  fftw_complex *cbuf ;
  
  if ( grbf_workspace_dimension(w) != 3 ) {
    g_error("%s: workplace dimension (%d) not equal to three",
	    __FUNCTION__, grbf_workspace_dimension(w)) ;
  }    

  Ef = w->Ef ; buf = w->buf ; cbuf = (fftw_complex *)(w->buf) ;
  for ( j = 0 ; j < nj ; j ++ ) {
    for ( k = 0 ; k < nk ; k ++ ) {
      for ( m = 0 ; m < nc ; m ++ ) {
	fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,0),
			     &(F[j*ldfj+k*fstr+m]), cbuf) ;
	vector_mul_complex(ni/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,0),
			     cbuf, &(wt[j*ldwj+k*wtstr+m])) ;
      }
    }
  }
  
#if 0
  /*test code for checking FFTs*/
  {gdouble err = 0 ;
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
	fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,1),
			     &(wt[i*ldwi+k*wtstr+m]), cbuf) ;
	vector_mul_complex(nj/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,1),
			     cbuf, &(wt[i*ldwi+k*wtstr+m])) ;
      }
    }
  }

#if 0
  /*test code for checking FFTs*/
  {gdouble err = 0 ;
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
	fftw_execute_dft_r2c(grbf_workspace_plan_forward(w,2),
			     &(wt[i*ldwi+j*ldwj+m]), cbuf) ;
	vector_mul_complex(nk/2+1, 1.0, buf, 1, Ef, 1) ;
	
	fftw_execute_dft_c2r(grbf_workspace_plan_inverse(w,2),
			     cbuf, &(wt[i*ldwi+j*ldwj+m])) ;
      }
    }
  }

#if 0
  /*test code for checking FFTs*/
  {gdouble err = 0 ;
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

grbf_workspace_t *grbf_workspace_alloc(gint dim, gint len)

{
  grbf_workspace_t *w ;

  w = (grbf_workspace_t *)g_malloc0(sizeof(grbf_workspace_t)) ;

  grbf_workspace_dimension(w) = dim ;

  if ( dim == 1 ) {
    grbf_workspace_length(w)    = len ;
    w->buf = (gdouble *)g_malloc(3*(len+2)*sizeof(gdouble)) ;
    w->Ef = &(w->buf[(len+2)]) ;
    return w ;
  }

  if ( dim == 2 ) {
    grbf_workspace_length(w) = len ;
    w->buf = (gdouble *)g_malloc(3*(len+2)*sizeof(gdouble)) ;
    w->Ef = &(w->buf[(len+2)]) ;

    return w ;
  }

  if ( dim == 3 ) {
    grbf_workspace_length(w) = len ;
    w->buf = (gdouble *)g_malloc(4*(len+2)*sizeof(gdouble)) ;
    w->Ef = &(w->buf[(len+2)]) ;

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
