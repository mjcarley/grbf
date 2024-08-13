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
