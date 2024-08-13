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

#ifndef __GRBF_H_INCLUDED__
#define __GRBF_H_INCLUDED__

#include <fftw3.h>

typedef struct _grbf_workspace_t grbf_workspace_t ;
struct _grbf_workspace_t {
  gint dim, len, N ;
  gdouble al, *buf, *Ef ;
  fftw_plan fp[3], ip[3] ;
} ;

#define grbf_workspace_dimension(_w)       ((_w)->dim)
#define grbf_workspace_length(_w)          ((_w)->len)
#define grbf_workspace_weight_number(_w)   ((_w)->N)
#define grbf_workspace_alpha(_w)           ((_w)->al)
#define grbf_workspace_plan_forward(_w,_i) ((_w)->fp[(_i)])
#define grbf_workspace_plan_inverse(_w,_i) ((_w)->ip[(_i)])

gint grbf_cardinal_function_coefficients(gdouble al, gint N,
					 gdouble *E, gboolean duplicate,
					 gdouble *rcond, gdouble *work) ;
gint grbf_interpolation_weights_1d_slow(gdouble *F, gint fstr, gint nf, gint nc,
					gdouble *E, gint ne,
					gboolean duplicated,
					gdouble *w, gint wstr) ;
gint grbf_interpolation_weights_1d_fft(gdouble *F, gint fstr, gint nf, gint nc,
				       gdouble *E, gint ne, gboolean duplicated,
				       gdouble *w, gint wstr) ;
gint grbf_interpolation_weight_1d(gdouble *F, gint fstr, gint nf, gint nc,
				  gdouble *E, gint N, gboolean duplicated,
				  gint i, gdouble *w) ;
gint grbf_interpolation_weights_2d_slow(gdouble *F, gint fstr, gint nf,
					gint ni, gint ldf, gint nj,
					gdouble *E, gint N, gboolean duplicated,
					gdouble *w, gint wstr, gint ldw) ;
gint grbf_interpolation_weights_3d(gdouble *F, gint fstr, gint nf,
				   gint ni, gint nj, gint nk,
				   gdouble *E, gint N, gboolean duplicated,
				   gdouble *w, gint wstr) ;
gint grbf_interpolation_eval_1d(gdouble al,
				gdouble *w, gint wstr, gint nw,
				gdouble x, gdouble *f) ;
gint grbf_interpolation_eval_2d(gdouble al,
				gdouble *w, gint wstr, gint nw,
				gint ni, gint ldw, gint nj,
				gdouble *x, gdouble *f) ;
gint grbf_interpolation_eval_3d(gdouble al,
				gdouble *w, gint wstr, gint nw,
				gint ni, gint nj, gint nk,
				gdouble *x, gdouble *f) ;
gint grbf_cardinal_interpolation_eval_1d(gdouble al,
					 gdouble *F, gint fstr, gint nf,
					 gint nc,
					 gdouble x, gdouble *f) ;
gint grbf_cardinal_interpolation_eval_2d(gdouble al,
					 gdouble *F, gint fstr,
					 gint ni, gint ldf, gint nj,
					 gint nc,
					 gdouble *x, gdouble *f) ;
gint grbf_cardinal_interpolation_eval_3d(gdouble al,
					 gdouble *F, gint fstr,
					 gint ni, gint nj, gint nk,
					 gint nc,
					 gdouble *x, gdouble *f) ;

gdouble grbf_cardinal_func(gdouble al, gdouble x) ;

gint grbf_gaussian_eval_1d(gdouble *y, gint ystr, gint nx,
			   gdouble *s, gint sstr,
			   gdouble *w, gint wstr, gint nf,
			   gdouble  x, gdouble *f) ;
gint grbf_gaussian_eval_2d(gdouble *y, gint ystr, gint ny,
			   gdouble *s, gint sstr,
			   gdouble *w, gint wstr, gint nf,
			   gdouble *x, gdouble *f) ;
gint grbf_gaussian_eval_3d(gdouble *y, gint ystr, gint ny,
			   gdouble *s, gint sstr,
			   gdouble *w, gint wstr, gint nf,
			   gdouble *x, gdouble *f) ;

gint grbf_interpolation_weights_fft_1d(gdouble *F, gint fstr, gint nf, gint nc,
				       grbf_workspace_t *w,
				       gdouble *wt, gint wtstr) ;
gint grbf_interpolation_weights_fft_2d(gdouble *F, gint fstr,
				       gint ni, gint ldf, gint nj, gint nc,
				       grbf_workspace_t *w,
				       gdouble *wt, gint wtstr, gint ldw) ;
gint grbf_interpolation_weights_fft_3d(gdouble *F, gint fstr,
				       gint ni, gint ldfi,
				       gint nj, gint ldfj,
				       gint nk, gint nc,
				       grbf_workspace_t *w,
				       gdouble *wt, gint wtstr,
				       gint ldwi, gint ldwj) ;

grbf_workspace_t *grbf_workspace_alloc(gint dim, gint len) ;
gint grbf_workspace_init_1d(grbf_workspace_t *w,
			    gdouble *F, gint fstr, gint nf,
			    gdouble *wt, gint wstr,
			    gint nc, gdouble al, gint N, gdouble *work) ;
gint grbf_workspace_init_2d(grbf_workspace_t *w,
			    gdouble *F, gint fstr, gint ni, gint ldf, gint nj,
			    gdouble *wt, gint wtstr, gint ldw,
			    gdouble al, gint N, gdouble *work) ;
gint grbf_workspace_init_3d(grbf_workspace_t *w,
			    gdouble *F, gint fstr,
			    gint ni, gint ldfi,
			    gint nj, gint ldfj,
			    gint nk,
			    gdouble *wt, gint wtstr,
			    gint ldwi, gint ldwj,
			    gdouble al, gint N, gdouble *work) ;

#endif /*__GRBF_H_INCLUDED__*/
