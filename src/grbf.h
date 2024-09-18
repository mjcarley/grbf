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

#define grbf_cardinal_function_length(_al,_tol)			\
  ((gint)ceil(-log(M_PI*0.5*(_tol)/((_al)*(_al)))/((_al)*(_al))))

typedef struct _grbf_workspace_t grbf_workspace_t ;
struct _grbf_workspace_t {
  gsize size ;
  gint dim, len, N ;
  gdouble al ;
  gpointer buf, Ef ;
  fftw_plan fp[3], ip[3] ;
  fftwf_plan fpf[3], ipf[3] ;
} ;

#define grbf_workspace_dimension(_w)         ((_w)->dim)
#define grbf_workspace_length(_w)            ((_w)->len)
#define grbf_workspace_data_size(_w)         ((_w)->size)
#define grbf_workspace_weight_number(_w)     ((_w)->N)
#define grbf_workspace_alpha(_w)             ((_w)->al)
#define grbf_workspace_plan_forward(_w,_i)   ((_w)->fp[(_i)])
#define grbf_workspace_plan_inverse(_w,_i)   ((_w)->ip[(_i)])
#define grbf_workspace_plan_forward_f(_w,_i) ((_w)->fpf[(_i)])
#define grbf_workspace_plan_inverse_f(_w,_i) ((_w)->ipf[(_i)])

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
			   gdouble *w, gint wstr, gint nc,
			   gdouble *x, gdouble *f) ;
gint grbf_gaussian_eval_3d(gdouble *y, gint ystr, gint ny,
			   gdouble *s, gint sstr,
			   gdouble *w, gint wstr, gint nf,
			   gdouble *x, gdouble *f) ;

gint grbf_grid_adjust_2d(gdouble *xmin, gdouble *xmax, gint *nx,
			 gdouble *ymin, gdouble *ymax, gint *ny,
			 gdouble del) ;
gint grbf_grid_adjust_3d(gdouble *xmin, gdouble *xmax, gint *nx,
			 gdouble *ymin, gdouble *ymax, gint *ny,
			 gdouble *zmin, gdouble *zmax, gint *nz,
			 gdouble del) ;
gint grbf_grid_count_points_2d(gdouble *F, gint fstr,
			       gint ni, gint ldf,
			       gint nj, gint nc,
			       gdouble tol) ;
gint grbf_grid_count_points_3d(gdouble *F, gint fstr,
			       gint ni, gint ldfi,
			       gint nj, gint ldfj,
			       gint nk, gint nc,
			       gdouble tol) ;

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

gint grbf_grid_increment_3d(gdouble *F, gint fstr,
			    gint ni, gint ldfi,
			    gint nj, gint ldfj,
			    gint nk, gint nc,
			    gdouble *origin, gdouble h,
			    gdouble *x, gdouble s, gdouble *q, gdouble tol) ;

gint grbf_cardinal_function_coefficients_f(gfloat al, gint N,
					 gfloat *E, gboolean duplicate,
					 gfloat *rcond, gfloat *work) ;
gint grbf_interpolation_weights_1d_slow_f(gfloat *F, gint fstr, gint nf, gint nc,
					gfloat *E, gint ne,
					gboolean duplicated,
					gfloat *w, gint wstr) ;
gint grbf_interpolation_weights_1d_fft_f(gfloat *F, gint fstr, gint nf, gint nc,
				       gfloat *E, gint ne, gboolean duplicated,
				       gfloat *w, gint wstr) ;
gint grbf_interpolation_weight_1d_f(gfloat *F, gint fstr, gint nf, gint nc,
				  gfloat *E, gint N, gboolean duplicated,
				  gint i, gfloat *w) ;
gint grbf_interpolation_weights_2d_slow_f(gfloat *F, gint fstr, gint nf,
					gint ni, gint ldf, gint nj,
					gfloat *E, gint N, gboolean duplicated,
					gfloat *w, gint wstr, gint ldw) ;
gint grbf_interpolation_weights_3d_f(gfloat *F, gint fstr, gint nf,
				   gint ni, gint nj, gint nk,
				   gfloat *E, gint N, gboolean duplicated,
				   gfloat *w, gint wstr) ;
gint grbf_interpolation_eval_1d_f(gfloat al,
				gfloat *w, gint wstr, gint nw,
				gfloat x, gfloat *f) ;
gint grbf_interpolation_eval_2d_f(gfloat al,
				gfloat *w, gint wstr, gint nw,
				gint ni, gint ldw, gint nj,
				gfloat *x, gfloat *f) ;
gint grbf_interpolation_eval_3d_f(gfloat al,
				gfloat *w, gint wstr, gint nw,
				gint ni, gint nj, gint nk,
				gfloat *x, gfloat *f) ;
gint grbf_cardinal_interpolation_eval_1d_f(gfloat al,
					 gfloat *F, gint fstr, gint nf,
					 gint nc,
					 gfloat x, gfloat *f) ;
gint grbf_cardinal_interpolation_eval_2d_f(gfloat al,
					 gfloat *F, gint fstr,
					 gint ni, gint ldf, gint nj,
					 gint nc,
					 gfloat *x, gfloat *f) ;
gint grbf_cardinal_interpolation_eval_3d_f(gfloat al,
					 gfloat *F, gint fstr,
					 gint ni, gint nj, gint nk,
					 gint nc,
					 gfloat *x, gfloat *f) ;

gfloat grbf_cardinal_func_f(gfloat al, gfloat x) ;

gint grbf_gaussian_eval_1d_f(gfloat *y, gint ystr, gint nx,
			   gfloat *s, gint sstr,
			   gfloat *w, gint wstr, gint nf,
			   gfloat  x, gfloat *f) ;
gint grbf_gaussian_eval_2d_f(gfloat *y, gint ystr, gint ny,
			   gfloat *s, gint sstr,
			   gfloat *w, gint wstr, gint nc,
			   gfloat *x, gfloat *f) ;
gint grbf_gaussian_eval_3d_f(gfloat *y, gint ystr, gint ny,
			   gfloat *s, gint sstr,
			   gfloat *w, gint wstr, gint nf,
			   gfloat *x, gfloat *f) ;

gint grbf_grid_adjust_2d_f(gfloat *xmin, gfloat *xmax, gint *nx,
			 gfloat *ymin, gfloat *ymax, gint *ny,
			 gfloat del) ;
gint grbf_grid_adjust_3d_f(gfloat *xmin, gfloat *xmax, gint *nx,
			 gfloat *ymin, gfloat *ymax, gint *ny,
			 gfloat *zmin, gfloat *zmax, gint *nz,
			 gfloat del) ;
gint grbf_grid_count_points_2d_f(gfloat *F, gint fstr,
			       gint ni, gint ldf,
			       gint nj, gint nc,
			       gfloat tol) ;
gint grbf_grid_count_points_3d_f(gfloat *F, gint fstr,
			       gint ni, gint ldfi,
			       gint nj, gint ldfj,
			       gint nk, gint nc,
			       gfloat tol) ;

gint grbf_interpolation_weights_fft_1d_f(gfloat *F, gint fstr, gint nf, gint nc,
				       grbf_workspace_t *w,
				       gfloat *wt, gint wtstr) ;
gint grbf_interpolation_weights_fft_2d_f(gfloat *F, gint fstr,
				       gint ni, gint ldf, gint nj, gint nc,
				       grbf_workspace_t *w,
				       gfloat *wt, gint wtstr, gint ldw) ;
gint grbf_interpolation_weights_fft_3d_f(gfloat *F, gint fstr,
				       gint ni, gint ldfi,
				       gint nj, gint ldfj,
				       gint nk, gint nc,
				       grbf_workspace_t *w,
				       gfloat *wt, gint wtstr,
				       gint ldwi, gint ldwj) ;

grbf_workspace_t *grbf_workspace_alloc_f(gint dim, gint len) ;
gint grbf_workspace_init_1d_f(grbf_workspace_t *w,
			    gfloat *F, gint fstr, gint nf,
			    gfloat *wt, gint wstr,
			    gint nc, gfloat al, gint N, gfloat *work) ;
gint grbf_workspace_init_2d_f(grbf_workspace_t *w,
			    gfloat *F, gint fstr, gint ni, gint ldf, gint nj,
			    gfloat *wt, gint wtstr, gint ldw,
			    gfloat al, gint N, gfloat *work) ;
gint grbf_workspace_init_3d_f(grbf_workspace_t *w,
			    gfloat *F, gint fstr,
			    gint ni, gint ldfi,
			    gint nj, gint ldfj,
			    gint nk,
			    gfloat *wt, gint wtstr,
			    gint ldwi, gint ldwj,
			    gfloat al, gint N, gfloat *work) ;

gint grbf_grid_increment_3d_f(gfloat *F, gint fstr,
			    gint ni, gint ldfi,
			    gint nj, gint ldfj,
			    gint nk, gint nc,
			    gfloat *origin, gfloat h,
			    gfloat *x, gfloat s, gfloat *q, gfloat tol) ;

#endif /*__GRBF_H_INCLUDED__*/
