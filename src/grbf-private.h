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

#ifndef __GRBF_PRIVATE_H_INCLUDED__
#define __GRBF_PRIVATE_H_INCLUDED__

#ifdef GRBF_SINGLE_PRECISION

#define GRBF_REAL gfloat

#define GRBF_FUNCTION_NAME(_func) _func##_f

#define SQRT(_x) sqrtf((_x))
#define CBRT(_x) cbrtf((_x))
#define SIN(_x) sinf((_x))
#define COS(_x) cosf((_x))
#define ACOS(_x) acosf((_x))
#define ATAN(_x) atanf((_x))
#define ATAN2(_y,_x) atan2f((_y),(_x))
#define LOG(_x) logf((_x))

#else

#define GRBF_REAL gdouble

#define GRBF_FUNCTION_NAME(_func) _func

#define SQRT(_x) sqrt((_x))
#define CBRT(_x) cbrt((_x))
#define SIN(_x) sin((_x))
#define COS(_x) cos((_x))
#define ACOS(_x) acos((_x))
#define ATAN(_x) atan((_x))
#define ATAN2(_y,_x) atan2((_y),(_x))
#define LOG(_x) log((_x))

#endif /*GRBF_SINGLE_PRECISION*/

#define grbf_vector1d_distance2(_x,_y)		\
  (((_x)[0]-(_y)[0])*((_x)[0]-(_y)[0]))
#define grbf_vector2d_distance2(_x,_y)		\
  (((_x)[0]-(_y)[0])*((_x)[0]-(_y)[0]) +	\
   ((_x)[1]-(_y)[1])*((_x)[1]-(_y)[1]))
#define grbf_vector3d_distance2(_x,_y)		\
  (((_x)[0]-(_y)[0])*((_x)[0]-(_y)[0]) +	\
   ((_x)[1]-(_y)[1])*((_x)[1]-(_y)[1]) +	\
   ((_x)[2]-(_y)[2])*((_x)[2]-(_y)[2]))

#endif /*__GRBF_PRIVATE_H_INCLUDED__*/
