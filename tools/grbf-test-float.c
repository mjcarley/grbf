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

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <glib.h>

#include <grbf.h>

char *progname ;
GTimer *timer ;

static gint parse_test(char *str)

{
  gint i ;
  char *tests[] = {"cardinal_func", "cardinal_coefficients",
    "interpolation", "interpolation_mapped", "interpolation_2d",
    "interpolation_mapped_2d", "interpolation_3d", "interpolation_mapped_3d",
    NULL} ;

  for ( i = 0 ; tests[i] != NULL ; i ++ ) {
    if ( strcmp(str, tests[i]) == 0) return i + 1 ;
  }
  
  return -1 ;
}

static void test_func(gint d, gfloat *x, gfloat *p, gint nf, gfloat *f)

{
  gfloat R2, R ;

  g_assert(nf > 0) ;
  
  switch ( d ) {
  default: g_assert_not_reached() ; break ;
  case 1:
    f[0] = sin(p[0]*(x[0]+p[1]*x[0]*x[0]))*exp(-p[2]*x[0]*x[0]) ;
    if ( nf == 1 ) return ;
    f[1] = cos(p[0]*(x[0]+p[1]*x[0]*x[0]))*exp(-p[2]*x[0]*x[0]) ;
    if ( nf == 2 ) return ;
    break ;
  case 2:
    R2 = x[0]*x[0] + x[1]*x[1] ;
    R = sqrt(R2) ;
    f[0] = cos(p[0]*R)*exp(-p[1]*R2) ;
    if ( nf == 1 ) return ;
    f[1] = cos(p[0]*R)*exp(-p[1]*R2*R2) ;
    break ;
  case 3:
    R2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] ;
    R = sqrt(R2) ;
    if ( nf == 3 ) {
      gfloat r, ph, th, r0, rh, s, z, w ;
      s = 0.25 ; r0 = 1.2 ;
      /*vortex ring test case*/
      th = atan2(x[1], x[0]) ;
      r = sqrt(x[0]*x[0] + x[1]*x[1]) ;
      z = x[2] ;
      rh = sqrt((r-r0)*(r-r0) + z*z) ;
      w = exp(-rh*rh/s/s) ;
      f[0] = -w*sin(th) ;
      f[1] = -w*cos(th) ;
      f[2] = 0 ;
      return ;
    }
    
    f[0] = cos(p[0]*R)*exp(-p[1]*R2) ;
    break ;
  }    
  
  return ;
}

static void cardinal_coefficients(gfloat al, gint N)

{
  gfloat *work, *E, rcond ;
  gint wsize, i ;

  fprintf(stderr, "coefficients of cardinal function\n") ;
  fprintf(stderr, "=================================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;
  
  wsize = (N+1)*(N+1) + 3*(N+1) ;
  E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ;
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  grbf_cardinal_function_coefficients_f(al, N, E, TRUE, &rcond, work) ;

  fprintf(stderr, "rcond = %lg\n", rcond) ;

  /* for ( i = 0 ; i <= N ; i ++ ) { */
  for ( i = 0 ; i <= 2*N ; i ++ ) {
    fprintf(stdout, "%d %1.16e\n", i, E[i]) ;
  }
  
  return ;
}  

static void interpolate_1d(gfloat al, gint N, gint nf)

{
  gfloat *work, *F, *w, x, y, g[32], p[32], errg[32], errc[32], ft[32] ;
  gint wsize, i, fstr, wstr, nc ;
  /* gboolean duplicate ; */
  grbf_workspace_t *ws ;
  
  fprintf(stderr, "interpolation test\n") ;
  fprintf(stderr, "==================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;
  fprintf(stderr, "%d points\n", nf) ;

  wsize = (N+1)*(N+1) ;
  /* E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ; */
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  p[0] = 0.5 ; p[1] = 0.002 ; p[2] = 0.0002 ;

  nc = 1 ;  
  fstr = 2 ; wstr = 2 ;
  F = (gfloat *)g_malloc0(fstr*(nf+2)*sizeof(gfloat)) ;
  w = (gfloat *)g_malloc0(wstr*(nf+2)*sizeof(gfloat)) ;

  ws = grbf_workspace_alloc_f(1, nf) ;
  grbf_workspace_init_1d_f(ws, F, fstr, nf, w, wstr, nc, al, N, work) ;
  
  /* duplicate = TRUE ; */
  /* grbf_cardinal_function_coefficients(al, N, E, duplicate, NULL, work) ; */

  for ( i = 0 ; i < nf ; i ++ ) {
    x = i - nf/2 ;
    test_func(1, &x, p, nc, &(F[i*fstr])) ;
  }
  
  /* grbf_interpolation_weights_1d_fft(F, fstr, nf, nc, E, N, duplicate, w, wstr) ; */

  grbf_interpolation_weights_fft_1d_f(F, fstr, nf, nc, ws, w, wstr) ;
  
  /* for ( x = nf/2-16 ; x < nf/2 + 16 ; x += 0.03 ) { */
  memset(errg, 0, nc*sizeof(gfloat)) ;
  memset(errc, 0, nc*sizeof(gfloat)) ;
  for ( x = 20 ; x < nf-20 ; x += 0.25 ) {
    y = x - nf/2 ;
    test_func(1, &y, p, nc, ft) ;
    memset(g, 0, nc*sizeof(gfloat)) ;
    grbf_interpolation_eval_1d_f(al, w, wstr, nf, x, g) ;

    for ( i = 0 ; i < nc ; i ++ ) {
      errg[i] = MAX(errg[i], fabs(ft[i]-g[i])) ;
    }
    
    fprintf(stdout, "%1.16e %1.16e", x, g[0]) ;
    memset(g, 0, nc*sizeof(gfloat)) ;
    grbf_cardinal_interpolation_eval_1d_f(al, F, fstr, nf, nc, x, g) ;
    for ( i = 0 ; i < nc ; i ++ ) {
      errc[i] = MAX(errc[i], fabs(ft[i]-g[i])) ;
    }
    
    fprintf(stdout, " %1.16e ", g[0]) ;
    fprintf(stdout, " %1.16e\n", ft[0]) ;
  }

  fprintf(stderr, "maximum error (interpolation)     = %lg\n", errg[0]) ;
  fprintf(stderr, "maximum error (cardinal)          = %lg\n", errc[0]) ;
  fprintf(stderr, "minimum weight                    = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)) ;
  
  return ;
}

static void interpolate_2d(gfloat al, gint N, gint nf)

{
  gfloat *work, *E, *F, *w, x[2], y[2], g[32], p[32], rcond, ft[32] ;
  gfloat errg[32], errc[32], t0 ;
  gint wsize, i, j, ni, nj, fstr, wstr, nc, ldf, ldw ;
  grbf_workspace_t *ws ;

  fprintf(stderr, "2D interpolation test\n") ;
  fprintf(stderr, "=====================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;
  
  wsize = (N+1)*(N+1) + 3*(N+1) ;
  E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ;
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  p[0] = 0.125 ; p[1] = 0.001 ; p[2] = 0.00125 ;
  
  fstr = 2 ; wstr = 2 ; nc = 2 ;
  ni = nj = nf ; nj -= 7 ;
  ldf = nj*fstr ; ldw = nj*wstr ;
  
  F = (gfloat *)g_malloc0(ni*ldf*sizeof(gfloat)) ;
  w = (gfloat *)g_malloc0(ni*ldw*sizeof(gfloat)) ;

  ws = grbf_workspace_alloc_f(2, ni) ;
  grbf_workspace_init_2d_f(ws, F, fstr, ni, ldf, nj, w, wstr, ldw,
			 al, N, work) ;
  
  fprintf(stderr, "%dx%d = %d points\n", ni, nj, ni*nj) ;

  grbf_cardinal_function_coefficients_f(al, N, E, TRUE, &rcond, work) ;

  fprintf(stderr, "rcond = %lg\n", rcond) ;

  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      x[0] = i-ni/2 ; x[1] = j - nj/2 ;
      test_func(2, x, p, nc, &(F[i*ldf+j*fstr])) ;
    }
  }

  /* fprintf(stderr, "evaluating Gaussian weights (direct) ") ; */
  /* t0 = g_timer_elapsed(timer, NULL) ; */
  /* grbf_interpolation_weights_2d_slow(F, fstr, nc, ni, ldf, nj, E, N, TRUE, */
  /* 				     w, wstr, ldw) ; */
  /* fprintf(stderr, "[%lg]\n", g_timer_elapsed(timer, NULL)-t0) ; */
  fprintf(stderr, "evaluating Gaussian weights (FFT) ") ;
  t0 = g_timer_elapsed(timer, NULL) ;
  grbf_interpolation_weights_fft_2d_f(F, fstr, ni, ldf, nj, nc, ws, w, wstr, ldw);
  fprintf(stderr, "[%lg]\n", g_timer_elapsed(timer, NULL)-t0) ;

  /* for ( x = nf/2-16 ; x < nf/2 + 16 ; x += 0.03 ) { */
  memset(errg, 0, nc*sizeof(gfloat)) ;
  memset(errc, 0, nc*sizeof(gfloat)) ;
  for ( x[0] = 30 ; x[0] < ni-30 ; x[0] += 1.5 ) {
    for ( x[1] = 30 ; x[1] < nj-30 ; x[1] += 1.5 ) {
      y[0] = x[0] - ni/2 ; y[1] = x[1] - nj/2 ;
      test_func(2, y, p, nc, ft) ;
      memset(g, 0, nc*sizeof(gfloat)) ;
      grbf_interpolation_eval_2d_f(al, w, wstr, nc, ni, ldw, nj, x, g) ;
      for ( i = 0 ; i < nc ; i ++ )  errg[i] = MAX(errg[i], fabs(g[i]-ft[i])) ;
      fprintf(stdout, "%1.16e %1.16e %1.16e", x[0], x[1], g[0]) ;
      memset(g, 0, nc*sizeof(gfloat)) ;
      grbf_cardinal_interpolation_eval_2d_f(al, F, fstr, ni, ldf, nj, nc, x, g) ;
      for ( i = 0 ; i < nc ; i ++ )  errc[i] = MAX(errc[i], fabs(g[i]-ft[i])) ;
      fprintf(stdout, " %1.16e %1.16e\n", g[0], ft[0]) ;
    }
  }

  fprintf(stderr, "maximum error (interpolation)     =") ;
  for ( i = 0 ; i < nc ; i ++ ) fprintf(stderr, " %lg", errg[i]) ;
  fprintf(stderr, "\n") ;
  fprintf(stderr, "maximum error (cardinal)          =") ;
  for ( i = 0 ; i < nc ; i ++ ) fprintf(stderr, " %lg", errc[i]) ;
  fprintf(stderr, "\n") ;
  fprintf(stderr, "minimum weight                    = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)) ;
  fprintf(stderr, "expected error (w_min/rcond)      = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)/rcond) ;
  fprintf(stderr, "expected error (cardinal/rcond)   = %lg\n",
	  errc[0]/rcond) ;
  
  return ;
}

static void interpolate_3d(gfloat al, gint N, gint nf)

{
  gfloat *work, *E, *F, *w, x[3], y[3], g, p[32], rcond, ft, errg, errc ;
  gfloat t0 ;
  gint wsize, i, j, k, ni, nj, nk, fstr, wstr, nc, ldfi, ldfj, ldwi, ldwj ;
  grbf_workspace_t *ws ;

  fprintf(stderr, "3D interpolation test\n") ;
  fprintf(stderr, "=====================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;

  wsize = (N+1)*(N+1) + 3*(N+1) ;
  E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ;
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  p[0] = 0.25 ; p[1] = 0.002 ; p[2] = 0.00125 ;
  
  fstr = 1 ; wstr = 1 ; nc = 1 ;
  ni = nj = nk = nf ;
  ldfj = nk*fstr ; ldfi = nj*ldfj ;
  ldwj = nk*wstr ; ldwi = nj*ldwj ;
  F = (gfloat *)g_malloc0(ni*ldfi*sizeof(gfloat)) ;
  w = (gfloat *)g_malloc0(ni*ldwi*sizeof(gfloat)) ;
  
  fprintf(stderr, "%dx%dx%d=%d points\n", ni, nj, nk, ni*nj*nk) ;

  grbf_cardinal_function_coefficients_f(al, N, E, TRUE, &rcond, work) ;

  fprintf(stderr, "rcond = %lg\n", rcond) ;
  
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      for ( k = 0 ; k < nk ; k ++ ) {
	x[0] = i - ni/2 ; x[1] = j - nj/2 ; x[2] = k - nk/2 ;
	test_func(3, x, p, nc, &(F[i*ldfi+j*ldfj+k*fstr])) ;
      }
    }
  }

  i = ni/2 ; j = nj/2 ; k = nk/2 ;
  /* fprintf(stderr, "generating interpolation weights\n") ; */
  /* grbf_interpolation_weights_3d(F, fstr, nc, ni, nj, nk, E, N, TRUE, w, wstr) ; */

  /* fprintf(stderr, "wt (slow) = %lg\n", w[i*ni*nj+j*nk+k]) ; */
  
  ws = grbf_workspace_alloc(3, MAX(ni,MAX(nj,nk))) ;
  grbf_workspace_init_3d_f(ws, F, fstr, ni, ldfi, nj, ldfj, nk,
			       w, wstr, ldwi, ldwj, al, N, work) ;

  fprintf(stderr, "evaluating Gaussian weights (FFT) ") ;
  t0 = g_timer_elapsed(timer, NULL) ;
  grbf_interpolation_weights_fft_3d_f(F, fstr, ni, ldfi, nj, ldfj, nk, nc,
					  ws, w, wstr, ldwi, ldwj);
  fprintf(stderr, "[%lg]\n", g_timer_elapsed(timer, NULL)-t0) ;
  
  /* for ( x = nf/2-16 ; x < nf/2 + 16 ; x += 0.03 ) { */
  errc = errg = 0.0 ;
  fprintf(stderr, "evaluating field\n") ;
  for ( x[0] = ni/2-10 ; x[0] <= ni/2+10 ; x[0] += 1.5 ) {
    for ( x[1] = nj/2-10 ; x[1] <= nj/2+10 ; x[1] += 1.5 ) {
      for ( x[2] = nk/2-10 ; x[2] <= nk/2+10 ; x[2] += 1.5 ) {
	y[0] = x[0] - ni/2 ; 
	y[1] = x[1] - nj/2 ; 
	y[2] = x[2] - nk/2 ; 
	test_func(3, y, p, nc, &ft) ;
	g = 0 ;
	grbf_interpolation_eval_3d_f(al, w, wstr, nc, ni, nj, nk, x, &g) ;
	errg = MAX(errg, fabs(g-ft)) ;
	fprintf(stdout, "%1.16e %1.16e %1.16e %1.16e", x[0], x[1], x[2], g) ;
	g = 0 ;
	grbf_cardinal_interpolation_eval_3d_f(al, F, fstr, ni, nj, nk, nc,
						  x, &g) ;
	errc = MAX(errc, fabs(g-ft)) ;      
	fprintf(stdout, " %1.16e %1.16e\n", g, ft) ;
      }
    }
  }

  fprintf(stderr, "maximum error (interpolation)     = %lg\n", errg) ;
  fprintf(stderr, "maximum error (cardinal)          = %lg\n", errc) ;
  fprintf(stderr, "minimum weight                    = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)) ;
  fprintf(stderr, "expected error (w_min/rcond)      = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)/rcond) ;
  fprintf(stderr, "expected error (cardinal/rcond)   = %lg\n",
	  errc/rcond) ;
  
  return ;
}

static void mapping_3d(gfloat al, gint N, gint nf)

{
  gfloat *work, *E, *F, *w, x[3], *y ;
  gfloat g[32], p[32], rcond, ft[32], errg[32], errc[32] ;
  gfloat t0, x0, y0, z0, x1, y1, z1, s, dx ;
  gint wsize, i, j, k, ni, nj, nk, fstr, wstr, nc, ldfi, ldfj, ldwi, ldwj ;
  grbf_workspace_t *ws ;

  fprintf(stderr,
	  "three-dimensional mapped interpolation (non-unit spacing)\n") ;
  fprintf(stderr,
	  "=========================================================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;

  wsize = (N+1)*(N+1) + 3*(N+1) ;
  E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ;
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  p[0] = 0.25 ; p[1] = 0.002 ; p[2] = 0.00125 ;

  x0 = -2.0 ; x1 = 2.5 ; 
  y0 = -2.0 ; y1 = 2.5 ; 
  z0 = -1 ; z1 = 1.2 ; 

  ni = nf ;
  dx = (x1 - x0)/ni ; s = dx/al ;
  fprintf(stderr, "sigma = %lg\n", s) ;

  grbf_grid_adjust_3d_f(&x0, &x1, &ni, &y0, &y1, &nj, &z0, &z1, &nk, dx) ;

  fprintf(stderr, "limits x: %lg %lg (%d)\n", x0, x1, ni) ;
  fprintf(stderr, "limits y: %lg %lg (%d)\n", y0, y1, nj) ;
  fprintf(stderr, "limits z: %lg %lg (%d)\n", z0, z1, nk) ;
    
  fstr = 3 ; wstr = 3 ; nc = 3 ;
  ldfj = nk*fstr ; ldfi = nj*ldfj ;
  ldwj = nk*wstr ; ldwi = nj*ldwj ;
  F = (gfloat *)g_malloc0(ni*ldfi*sizeof(gfloat)) ;
  w = (gfloat *)g_malloc0(ni*ldwi*sizeof(gfloat)) ;
  y = (gfloat *)g_malloc0(3*ni*nj*nk*sizeof(gfloat)) ;

  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      for ( k = 0 ; k < nk ; k ++ ) {
	y[3*(nj*nk*i+nk*j+k) + 0] = x0 + i*dx ;
	y[3*(nj*nk*i+nk*j+k) + 1] = y0 + j*dx ;
	y[3*(nj*nk*i+nk*j+k) + 2] = z0 + k*dx ;
	test_func(3, &(y[3*(nj*nk*i+nk*j+k) + 0]), p, nc,
		  &(F[i*ldfi+j*ldfj+k*fstr])) ;
      }
    }
  }
  
  ws = grbf_workspace_alloc(3, MAX(ni,MAX(nj,nk))) ;
  grbf_workspace_init_3d_f(ws, F, fstr, ni, ldfi, nj, ldfj, nk,
			 w, wstr, ldwi, ldwj, al, N, work) ;

  fprintf(stderr, "evaluating Gaussian weights (FFT) ") ;
  t0 = g_timer_elapsed(timer, NULL) ;
  grbf_interpolation_weights_fft_3d_f(F, fstr, ni, ldfi, nj, ldfj, nk, nc,
				    ws, w, wstr, ldwi, ldwj);
  fprintf(stderr, "[%lg]\n", g_timer_elapsed(timer, NULL)-t0) ;

  /* return ; */
  
  /* for ( x = nf/2-16 ; x < nf/2 + 16 ; x += 0.03 ) { */

  memset(errg, 0, 3*sizeof(gfloat)) ;
  memset(errc, 0, 3*sizeof(gfloat)) ;
  fprintf(stderr, "evaluating field\n") ;

  for ( i = 0 ; i < 237 ; i ++ ) {
    x[1] = 0 ; x[2] = 0 ;
    x[0] = x0 + (x1 - x0)*i/237 ; 
    x[1] = y0 + (y1 - y0)*i/237 ; 
    x[2] = z0 + (z1 - z0)*i/237 ; 
    test_func(3, x, p, nc, ft) ;
    memset(g, 0, 3*sizeof(gfloat)) ;
    grbf_gaussian_eval_3d_f(y, 3, ni*nj*nk, &s, 0, w, wstr, nc,x, g) ;
    for ( j = 0 ; j < nc ; j ++ ) {
      errg[j] = MAX(errg[j], fabs(g[j]-ft[j])) ;
    }
    fprintf(stdout, "%1.16e %1.16e %1.16e %1.16e %1.16e %1.16e",
	    x[0], x[1], x[2], g[0], g[1], g[2]) ;
    /* g = 0 ; */
    /* grbf_cardinal_interpolation_eval_3d(al, F, fstr, ni, nj, nk, nc, x, &g) ; */
    /* errc = MAX(errc, fabs(g-ft)) ;       */
    fprintf(stdout, " %1.16e %1.16e %1.16e", ft[0], ft[1], ft[2]) ;
    fprintf(stdout, "\n") ;
  }

  fprintf(stderr, "maximum error (interpolation)     = %lg\n",
	  MAX(errg[0],MAX(errg[1],errg[2]))) ;
  /* fprintf(stderr, "maximum error (cardinal)          = %lg\n", errc) ; */
  fprintf(stderr, "minimum weight                    = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)) ;
  /* fprintf(stderr, "expected error (w_min/rcond)      = %lg\n", */
  /* 	  2.0*al*al/M_PI*exp(-al*al*N)/rcond) ; */
  /* fprintf(stderr, "expected error (cardinal/rcond)   = %lg\n", */
  /* 	  errc/rcond) ; */
  
  return ;
}

static void cardinal_func(gfloat al, gint N)

{
  gfloat *work, *E, f, x, rcond ;
  gint wsize, j ;

  fprintf(stderr, "evaluation of cardinal function\n") ;
  fprintf(stderr, "===============================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;

  wsize = (N+1)*(N+1) + 3*(N+1) ;
  E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ;
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  grbf_cardinal_function_coefficients_f(al, N, E, TRUE, &rcond, work) ;

  fprintf(stderr, "rcond = %lg\n", rcond) ;

  for ( x = -5 ; x <= 5 ; x += 0.1 ) {
    fprintf(stdout, "%1.16e %1.16e", x, grbf_cardinal_func_f(al, x)) ;
    f = 0 ;
    for ( j = 0 ; j < 2*N+1 ; j ++ )
      f += E[j]*exp(-al*al*(x-(j-N))*(x-(j-N))) ;
    fprintf(stdout, " %1.16e\n", f) ;
  }
  
  return ;
}

static void mapping_1d(gfloat al, gint N, gint nf)

{
  gfloat *work, *E, *F, *w, *y, x, ft[32], g[32], p[32], x0, x1, dx, s ;
  gfloat errs, errc[32], errg[32], rcond ;
  gint wsize, i, j, fstr, wstr, nc ;

  fprintf(stderr, "mapped interpolation (non-unit spacing)\n") ;
  fprintf(stderr, "=======================================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;

  errs = 4.0*exp(-M_PI*M_PI/al/al) ;
  
  wsize = (N+1)*(N+1) + 3*(N+1) ;
  E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ;
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  p[0] = 4.3 ; p[1] = 0.02 ; p[2] = 0.1 ;
  
  fstr = 2 ; wstr = 2 ; nc = 2 ;
  F = (gfloat *)g_malloc0(fstr*nf*sizeof(gfloat)) ;
  y = (gfloat *)g_malloc0(     nf*sizeof(gfloat)) ;
  w = (gfloat *)g_malloc0(wstr*nf*sizeof(gfloat)) ;
  
  grbf_cardinal_function_coefficients_f(al, N, E, TRUE, &rcond, work) ;

  fprintf(stderr, "rcond = %lg\n", rcond) ;
  
  x0 = -3 ; x1 = 5 ; 

  dx = (x1 - x0)/nf ; s = dx/al ;
  fprintf(stderr, "sigma = %lg\n", s) ;
  for ( i = 0 ; i < nf ; i ++ ) {
    y[i] = x0 + i*dx ;
    test_func(1, &(y[i]), p, nc, &(F[i*fstr])) ;
  }
  
  grbf_interpolation_weights_1d_slow_f(F, fstr, nf, nc, E, N, TRUE, w, wstr) ;

  memset(errg, 0, nc*sizeof(gfloat)) ;
  memset(errc, 0, nc*sizeof(gfloat)) ;
  for ( x = x0+2 ; x <= x1-2 ; x += 0.125 ) {
    test_func(1, &x, p, nc, ft) ;
    memset(g, 0, nc*sizeof(gfloat)) ;
    grbf_gaussian_eval_1d_f(y, 1, nf, &s, 0, w, wstr, nc, x, g) ;
    fprintf(stdout, "%1.16e", x) ;
    for ( j = 0 ; j < nc ; j ++ ) {
      errg[j] = MAX(errg[j], fabs(g[j]-ft[j])) ;
      fprintf(stdout, " %1.16e", g[j]) ;
    }
    memset(g, 0, nc*sizeof(gfloat)) ;
    grbf_cardinal_interpolation_eval_1d_f(al, F, fstr, nf, nc,
					(x-x0)/dx, g) ;
    /* fprintf(stdout, " %1.16e %1.16e\n", g, ft) ; */
    for ( j = 0 ; j < nc ; j ++ ) {
      errc[j] = MAX(errc[j], fabs(g[j]-ft[j])) ;
      fprintf(stdout, " %1.16e", g[j]) ;
    }
    for ( j = 0 ; j < nc ; j ++ ) {
      fprintf(stdout, " %1.16e", ft[j]) ;
    }
    fprintf(stdout, "\n") ;
  }

  fprintf(stderr, "maximum error (Gaussian) =") ;
  for ( j = 0 ; j < nc ; j ++ ) fprintf(stderr, " %lg", errg[j]) ;
  fprintf(stderr, "\n") ;
  fprintf(stderr, "maximum error (cardinal) =") ;
  for ( j = 0 ; j < nc ; j ++ ) fprintf(stderr, " %lg", errc[j]) ;
  fprintf(stderr, "\n") ;
  fprintf(stderr, "saturation error         = %lg\n", errs) ;
  fprintf(stderr, "minimum weight           = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)) ;
  fprintf(stderr, "expected error           = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)/rcond) ;
  /* fprintf(stderr, "expected error           = %lg\n", */
  /* 	  errc/rcond) ; */

  return ;
}

static void mapping_2d(gfloat al, gint N, gint nf)

{
  gfloat *work, *E, *F, *w, *y, x[2], g, p[32], x0, x1, y0, y1, dx, s ;
  gfloat errs, errc, errg, rcond, ft, ij[2], t0 ;
  gint wsize, i, j, fstr, wstr, nc, ni, nj ;

  fprintf(stderr, "two-dimensional mapped interpolation (non-unit spacing)\n") ;
  fprintf(stderr, "=======================================================\n") ;
  fprintf(stderr, "N     = %d\n", N) ;
  fprintf(stderr, "alpha = %lg\n", al) ;
  errs = 4.0*exp(-M_PI*M_PI/al/al) ;

  ni = nj = nf ;
  fprintf(stderr, "%dx%d=%d interpolation nodes\n", ni, nj, ni*nj) ;
  
  wsize = (N+1)*(N+1) + 3*(N+1) ;
  E = (gfloat *)g_malloc0((2*N+1)*sizeof(gfloat)) ;
  work = (gfloat *)g_malloc0(wsize*sizeof(gfloat)) ;

  p[0] = 4.3 ; p[1] = 0.02 ; p[2] = 0.1 ;
  
  fstr = 1 ; wstr = 1 ; nc = 1 ;
  F = (gfloat *)g_malloc0(fstr*(ni*nj)*sizeof(gfloat)) ;
  y = (gfloat *)g_malloc0(2*   (ni*nj)*sizeof(gfloat)) ;
  w = (gfloat *)g_malloc0(wstr*(ni*nj)*sizeof(gfloat)) ;
  
  grbf_cardinal_function_coefficients_f(al, N, E, TRUE, &rcond, work) ;

  fprintf(stderr, "rcond = %lg\n", rcond) ;
  
  x0 = -3 ; x1 = 5 ; 
  y0 = -3 ; y1 = 5 ; 

  dx = (x1 - x0)/ni ; s = dx/al ;
  fprintf(stderr, "sigma = %lg\n", s) ;
  for ( i = 0 ; i < ni ; i ++ ) {
    for ( j = 0 ; j < nj ; j ++ ) {
      y[2*(i*nj+j)+0] = x0 + i*dx ;
      y[2*(i*nj+j)+1] = y0 + j*dx ;
      test_func(2, &(y[2*(i*nj+j)]), p, nc, &(F[(i*nj+j)*fstr])) ;
    }
  }

  fprintf(stderr, "evaluating Gaussian weights ") ;
  t0 = g_timer_elapsed(timer, NULL) ;
  grbf_interpolation_weights_2d_slow_f(F, fstr, nc, ni, nj, nj, E, N, TRUE,
					   w, wstr, nj) ;
  fprintf(stderr, "[%lg]\n", g_timer_elapsed(timer, NULL)-t0) ;
  
  fprintf(stderr, "evaluating interpolant (cardinal function) ") ;
  t0 = g_timer_elapsed(timer, NULL) ;
  for ( x[0] = x0+2 ; x[0] <= x1-2 ; x[0] += 0.125 ) {
    for ( x[1] = y0+2 ; x[1] <= y1-2 ; x[1] += 0.125 ) {
      grbf_cardinal_interpolation_eval_2d_f(al, F, fstr, ni, nj, nj, nc, ij, &g) ;
    }
  }
  fprintf(stderr, "[%lg]\n", g_timer_elapsed(timer, NULL)-t0) ;

  fprintf(stderr, "evaluating interpolant (Gaussians) ") ;
  t0 = g_timer_elapsed(timer, NULL) ;
  for ( x[0] = x0+2 ; x[0] <= x1-2 ; x[0] += 0.125 ) {
    for ( x[1] = y0+2 ; x[1] <= y1-2 ; x[1] += 0.125 ) {
      grbf_gaussian_eval_2d_f(y, 2, ni*nj, &s, 0, w, wstr, nc, x, &g) ;
    }
  }
  fprintf(stderr, "[%lg]\n", g_timer_elapsed(timer, NULL)-t0) ;

  errc = errg = 0.0 ;
  for ( x[0] = x0+2 ; x[0] <= x1-2 ; x[0] += 0.125 ) {
    for ( x[1] = y0+2 ; x[1] <= y1-2 ; x[1] += 0.125 ) {
      test_func(2, x, p, nc, &ft) ;      
      g = 0 ;
      grbf_gaussian_eval_2d_f(y, 2, ni*nj, &s, 0, w, wstr, nc, x, &g) ;
      errg = MAX(errg, fabs(g-ft)) ;
      fprintf(stdout, "%1.16e %1.16e %1.16e", x[0], x[1], g) ;
      g = 0 ;
      ij[0] = (x[0] - x0)/dx ; ij[1] = (x[1] - y0)/dx ; 
      grbf_cardinal_interpolation_eval_2d_f(al, F, fstr, ni, nj, nj, nc, ij, &g) ;
      fprintf(stdout, " %1.16e %1.16e\n", g, ft) ;
      errc = MAX(errc, fabs(g-ft)) ;
    }
  }

  fprintf(stderr, "maximum error (Gaussian)          = %lg\n", errg) ;
  fprintf(stderr, "maximum error (cardinal)          = %lg\n", errc) ;
  fprintf(stderr, "saturation error                  = %lg\n", errs) ;
  fprintf(stderr, "minimum weight                    = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)) ;
  fprintf(stderr, "expected error (w_min/rcond)      = %lg\n",
	  2.0*al*al/M_PI*exp(-al*al*N)/rcond) ;
  fprintf(stderr, "expected error (cardinal/rcond)   = %lg\n",
	  errc/rcond) ;
  
  return ;
}

gint main(gint argc, char **argv)

{
  gfloat al, tol ;
  gint N, test, nf ;
  char ch ;

  progname = g_strdup(g_path_get_basename(argv[0])) ;
  timer = g_timer_new() ;

  al = 0.5 ; tol = 1e-9 ;

  test = 0 ; N = -1 ; nf = 256 ;
  
  while ( (ch = getopt(argc, argv, "a:e:N:n:t:")) != EOF ) {
    switch (ch ) {
    default: g_assert_not_reached() ; break ;
    case 'a': al = atof(optarg) ; break ;
    case 'e': tol = atof(optarg) ; break ;
    case 'N': N  = atoi(optarg) ; break ;
    case 'n': nf = atoi(optarg) ; break ;
    case 't': test = parse_test(optarg) ; break ;
    }
  }

  if ( N == -1 ) {
    /* N = (gint)ceil(-log(M_PI*tol/2.0/al/al)/al/al) ; */
    N = grbf_cardinal_function_length(al,tol) ;
  }
  
  if ( test == 0 ) {
    fprintf(stderr, "%s: no test specified\n", progname) ;

    return 0 ;
  }

  if ( test == -1 ) {
    fprintf(stderr, "%s: test not recognised\n", progname) ;

    return 0 ;
  }
  
  if ( test == 1 ) {
    cardinal_func(al, N) ;

    return 0 ;
  }

  if ( test == 2 ) {
    cardinal_coefficients(al, N) ;
    return 0 ;
  }

  if ( test == 3 ) {
    interpolate_1d(al, N, nf) ;

    return 0 ;
  }

  if ( test == 4 ) {
    mapping_1d(al, N, nf) ;

    return 0 ;
  }

  if ( test == 5 ) {
    interpolate_2d(al, N, nf) ;

    return 0 ;
  }
  
  if ( test == 6 ) {
    mapping_2d(al, N, nf) ;

    return 0 ;
  }

  if ( test == 7 ) {
    interpolate_3d(al, N, nf) ;

    return 0 ;
  }

  if ( test == 8 ) {
    mapping_3d(al, N, nf) ;

    return 0 ;
  }
  
  return 0 ;
}
