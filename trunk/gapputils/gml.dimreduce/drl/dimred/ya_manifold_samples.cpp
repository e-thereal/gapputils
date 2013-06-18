/**************************************************************************
                           ya_manifold_samples.cpp
                             -------------------
                                 Haixia Jia

    Sample manifolds for dimensionality reduction

 __________________________________________________________________________
    This file is part of the Dr.L. Library
__________________________________________________________________________

    begin      : Dec 1 2008
    authors    : Haixia Jia
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_MANIFOLD_SAMPLES_HPP
#define YA_MANIFOLD_SAMPLES_HPP

YALANAMESPACE

// Generate a japanese flag roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_japaneseflag(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng){
  eltype * x1 = new eltype[sample_size];
  eltype * x2 = new eltype[sample_size];
  eltype tem1, tem2;
  for (int i=0; i<sample_size; i++){
    do {
      tem1 = rng.rand01() * 2.0 - 1.0;
      tem2 = rng.rand01() * 2.0 - 1.0;
    }
    while ((tem1*tem1 + tem2*tem2) < 0.5);
    x1[i] = tem1;
    x2[i] = tem2;
    if (x1[i]<(-0.8)) colors[i] = 1;
    else if (x1[i]<(-0.6)) colors[i] = 2;
    else if (x1[i]<(-0.4))  colors[i] = 3;
    else if (x1[i]<(-0.2))  colors[i] = 4;
    else if (x1[i]<(0.0))  colors[i] = 5;
    else if (x1[i]<(0.2))  colors[i] = 6;
    else if (x1[i]<(0.4))  colors[i] = 7;
    else if (x1[i]<(0.6))  colors[i] = 8;
    else if (x1[i]<(0.8))  colors[i] = 9;
    else if (x1[i]<(1.0))  colors[i] = 10;
  }
  input.setup(sample_size,3);
  for (int i=0; i<sample_size; i++) {
    input(i, 0) = sqrt(2.0 + 2.0*x1[i]) * cos(2.0*PI*sqrt(2.0+2.0*x1[i]));
    input(i, 1) = sqrt(2.0 + 2.0*x1[i]) * sin(2.0*PI*sqrt(2.0+2.0*x1[i]));
    input(i, 2) = 2.0 * x2[i];
  }
  delete [] x1;
  delete [] x2;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}


// Generate a non developable swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_nondevroll(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng){
  eltype * x1 = new eltype[sample_size];
  eltype * x2 = new eltype[sample_size];
  for (int i=0; i<sample_size; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    if (x1[i]<(-0.8)) colors[i] = 1;
    else if (x1[i]<(-0.6)) colors[i] = 2;
    else if (x1[i]<(-0.4))  colors[i] = 3;
    else if (x1[i]<(-0.2))  colors[i] = 4;
    else if (x1[i]<(0.0))  colors[i] = 5;
    else if (x1[i]<(0.2))  colors[i] = 6;
    else if (x1[i]<(0.4))  colors[i] = 7;
    else if (x1[i]<(0.6))  colors[i] = 8;
    else if (x1[i]<(0.8))  colors[i] = 9;
    else if (x1[i]<(1.0))  colors[i] = 10;
  }
  input.setup(sample_size,3);
  for (int i=0; i<sample_size; i++) {
    input(i, 0) = (1 + x2[i] * x2[i]) * sqrt(1.0 + x1[i]) * cos(2.0*PI*sqrt(1.0+x1[i]));
    input(i, 1) = (1 + x2[i] * x2[i]) * sqrt(1.0 + x1[i]) * sin(2.0*PI*sqrt(1.0+x1[i]));
    input(i, 2) = x2[i] * 2.0;
  }
  delete [] x1;
  delete [] x2;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}


// Generate a sphere input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_sphere(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng){
  eltype * x1 = new eltype[sample_size];
  eltype * x2 = new eltype[sample_size];
  for (int i=0; i<sample_size; i++){
    x1[i] = rng.rand01() * PI;
    x2[i] = rng.rand01() * 2.0 * PI;
    eltype pp = PI / 10.0;
    if (x1[i]<(pp)) colors[i] = 1;
    else if (x1[i]<(pp*2.0)) colors[i] = 2;
    else if (x1[i]<(pp*3.0))  colors[i] = 3;
    else if (x1[i]<(pp*4.0))  colors[i] = 4;
    else if (x1[i]<(pp*5.0))  colors[i] = 5;
    else if (x1[i]<(pp*6.0))  colors[i] = 6;
    else if (x1[i]<(pp*7.0))  colors[i] = 7;
    else if (x1[i]<(pp*8.0))  colors[i] = 8;
    else if (x1[i]<(pp*9.0))  colors[i] = 9;
    else if (x1[i]<(pp*10.0))  colors[i] = 10;
  }
  input.setup(sample_size,3);
  for (int i=0; i<sample_size; i++) {
    input(i, 0) = cos(x1[i]) * cos(x2[i]);
    input(i, 1) = sin(x1[i]) * cos(x2[i]);
    input(i, 2) = sin(x2[i]);
  }
  delete [] x1;
  delete [] x2;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}


// Generate a torus input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_torus(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng){
  eltype * x1 = new eltype[sample_size];
  eltype * x2 = new eltype[sample_size];
  for (int i=0; i<sample_size; i++){
    x1[i] = rng.rand01() * 2.0 * PI;
    x2[i] = rng.rand01() * 2.0 * PI;
    eltype pp = 2.0 * PI / 10.0;
    if (x1[i]<(pp)) colors[i] = 1;
    else if (x1[i]<(pp*2.0)) colors[i] = 2;
    else if (x1[i]<(pp*3.0))  colors[i] = 3;
    else if (x1[i]<(pp*4.0))  colors[i] = 4;
    else if (x1[i]<(pp*5.0))  colors[i] = 5;
    else if (x1[i]<(pp*6.0))  colors[i] = 6;
    else if (x1[i]<(pp*7.0))  colors[i] = 7;
    else if (x1[i]<(pp*8.0))  colors[i] = 8;
    else if (x1[i]<(pp*9.0))  colors[i] = 9;
    else if (x1[i]<(pp*10.0))  colors[i] = 10;
  }
  input.setup(sample_size,3);
  for (int i=0; i<sample_size; i++) {
    input(i, 0) = (2.0 + cos(x1[i])) * cos(x2[i]);
    input(i, 1) = (2.0 + cos(x1[i])) * sin(x2[i]);
    input(i, 2) = sin(x1[i]);
  }
  delete [] x1;
  delete [] x2;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}

// Generate a trefoil knot input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_tknot(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng){
  eltype * x = new eltype[sample_size];
  for (int i=0; i<sample_size; i++){
    x[i] = rng.rand01() * 2.0 * PI;
    eltype pp = 2.0 * PI / 10.0;
    if (x[i]<(pp)) colors[i] = 1;
    else if (x[i]<(pp*2.0)) colors[i] = 2;
    else if (x[i]<(pp*3.0))  colors[i] = 3;
    else if (x[i]<(pp*4.0))  colors[i] = 4;
    else if (x[i]<(pp*5.0))  colors[i] = 5;
    else if (x[i]<(pp*6.0))  colors[i] = 6;
    else if (x[i]<(pp*7.0))  colors[i] = 7;
    else if (x[i]<(pp*8.0))  colors[i] = 8;
    else if (x[i]<(pp*9.0))  colors[i] = 9;
    else if (x[i]<(pp*10.0))  colors[i] = 10;
  }
  input.setup(sample_size,3);
  for (int i=0; i<sample_size; i++) {
    input(i, 0) = 41.0 * cos(x[i]) - 18.0 * sin(x[i]) - 83.0 * cos(x[i]) 
                  - 83.0 * sin(2.0 * x[i]) - 11.0 * sin(3.0 * x[i]) + 27.0 * sin(3.0*x[i]);
    input(i, 1) = 36.0 * cos(x[i]) + 27.0 * sin(x[i]) - 113.0 * cos(2.0*x[i]) 
                  + 30.0 * sin(2.0 * x[i]) + 11.0 * cos(3.0 * x[i]) - 27.0 * sin(3.0*x[i]);
    input(i, 2) = 45.0 * sin(x[i]) - 30.0 * cos(2.0*x[i]) 
                  + 113.0 * sin(2.0 * x[i]) - 11.0 * cos(3.0 * x[i]) + 27.0 * sin(3.0*x[i]);
  }
  delete [] x;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}



// Generate a box input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_box(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng){
  eltype * x1 = new eltype[sample_size*6];
  eltype * x2 = new eltype[sample_size*6];
  ya_sizet rsize = sample_size*6;
  input.setup(rsize, 3);
  for (int i=0; i<sample_size; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    colors[i] = 1;
    input(i, 0) = x1[i];
    input(i, 1) = x2[i];
    input(i, 2) = -1.0;
  }
  for (int i=sample_size; i<sample_size*2; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    input(i, 0) = x1[i];
    input(i, 1) = -1.0;
    input(i, 2) = x2[i];
    if (x2[i]<(-0.8)) colors[i] = 1;
    else if (x2[i]<(-0.6)) colors[i] = 2;
    else if (x2[i]<(-0.4))  colors[i] = 3;
    else if (x2[i]<(-0.2))  colors[i] = 4;
    else if (x2[i]<(0.0))  colors[i] = 5;
    else if (x2[i]<(0.2))  colors[i] = 6;
    else if (x2[i]<(0.4))  colors[i] = 7;
    else if (x2[i]<(0.6))  colors[i] = 8;
    else if (x2[i]<(0.8))  colors[i] = 9;
    else if (x2[i]<(1.0))  colors[i] = 10;
  }
  for (int i=sample_size*2; i<sample_size*3; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    input(i, 0) = x1[i];
    input(i, 1) = 1.0;
    input(i, 2) = x2[i];
    if (x2[i]<(-0.8)) colors[i] = 1;
    else if (x2[i]<(-0.6)) colors[i] = 2;
    else if (x2[i]<(-0.4))  colors[i] = 3;
    else if (x2[i]<(-0.2))  colors[i] = 4;
    else if (x2[i]<(0.0))  colors[i] = 5;
    else if (x2[i]<(0.2))  colors[i] = 6;
    else if (x2[i]<(0.4))  colors[i] = 7;
    else if (x2[i]<(0.6))  colors[i] = 8;
    else if (x2[i]<(0.8))  colors[i] = 9;
    else if (x2[i]<(1.0))  colors[i] = 10;
  }
  for (int i=sample_size*3; i<sample_size*4; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    input(i, 0) = 1.0;
    input(i, 1) = x1[i];
    input(i, 2) = x2[i];
    if (x2[i]<(-0.8)) colors[i] = 1;
    else if (x2[i]<(-0.6)) colors[i] = 2;
    else if (x2[i]<(-0.4))  colors[i] = 3;
    else if (x2[i]<(-0.2))  colors[i] = 4;
    else if (x2[i]<(0.0))  colors[i] = 5;
    else if (x2[i]<(0.2))  colors[i] = 6;
    else if (x2[i]<(0.4))  colors[i] = 7;
    else if (x2[i]<(0.6))  colors[i] = 8;
    else if (x2[i]<(0.8))  colors[i] = 9;
    else if (x2[i]<(1.0))  colors[i] = 10;
  }
  for (int i=sample_size*4; i<sample_size*5; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    input(i, 0) = -1.0;
    input(i, 1) = x1[i];
    input(i, 2) = x2[i];
    if (x2[i]<(-0.8)) colors[i] = 1;
    else if (x2[i]<(-0.6)) colors[i] = 2;
    else if (x2[i]<(-0.4))  colors[i] = 3;
    else if (x2[i]<(-0.2))  colors[i] = 4;
    else if (x2[i]<(0.0))  colors[i] = 5;
    else if (x2[i]<(0.2))  colors[i] = 6;
    else if (x2[i]<(0.4))  colors[i] = 7;
    else if (x2[i]<(0.6))  colors[i] = 8;
    else if (x2[i]<(0.8))  colors[i] = 9;
    else if (x2[i]<(1.0))  colors[i] = 10;
  }
  for (int i=sample_size*5; i<sample_size*6; i++){
    x1[i] = rng.rand01() * 2.0 + 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    colors[i] = 10;
    input(i, 0) = x1[i];
    input(i, 1) = x2[i];
    input(i, 2) = 1.0;
  }
  delete [] x1;
  delete [] x2;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}


// Generate a my swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_thinswissroll(YA_BaseT &input, int* colors,  ya_sizet sample_size, 
                   Rng &rng){
  eltype * x1 = new eltype[sample_size];
  eltype * x2 = new eltype[sample_size];
  for (int i=0; i<sample_size; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    if (x1[i]<(-0.8)) colors[i] = 1;
    else if (x1[i]<(-0.6)) colors[i] = 2;
    else if (x1[i]<(-0.4))  colors[i] = 3;
    else if (x1[i]<(-0.2))  colors[i] = 4;
    else if (x1[i]<(0.0))  colors[i] = 5;
    else if (x1[i]<(0.2))  colors[i] = 6;
    else if (x1[i]<(0.4))  colors[i] = 7;
    else if (x1[i]<(0.6))  colors[i] = 8;
    else if (x1[i]<(0.8))  colors[i] = 9;
    else if (x1[i]<(1.0))  colors[i] = 10;
  }
  input.setup(sample_size,3);
  for (int i=0; i<sample_size; i++) {
    input(i, 0) = sqrt(2.0 + 2.0*x1[i]) * cos(2.0*PI*sqrt(2.0+2.0*x1[i]));
    input(i, 1) = sqrt(2.0 + 2.0*x1[i]) * sin(2.0*PI*sqrt(2.0+2.0*x1[i]));
    input(i, 2) = x2[i] / 2.0;
  }
  delete [] x1;
  delete [] x2;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}

// Generate a swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_swissroll1(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng){
  eltype * x1 = new eltype[sample_size];
  eltype * x2 = new eltype[sample_size];
  for (int i=0; i<sample_size; i++){
    x1[i] = rng.rand01() * 2.0 - 1.0;
    x2[i] = rng.rand01() * 2.0 - 1.0;
    if (x1[i]<(-0.8)) colors[i] = 1;
    else if (x1[i]<(-0.6)) colors[i] = 2;
    else if (x1[i]<(-0.4))  colors[i] = 3;
    else if (x1[i]<(-0.2))  colors[i] = 4;
    else if (x1[i]<(0.0))  colors[i] = 5;
    else if (x1[i]<(0.2))  colors[i] = 6;
    else if (x1[i]<(0.4))  colors[i] = 7;
    else if (x1[i]<(0.6))  colors[i] = 8;
    else if (x1[i]<(0.8))  colors[i] = 9;
    else if (x1[i]<(1.0))  colors[i] = 10;
  }
  input.setup(sample_size,3);
  for (int i=0; i<sample_size; i++) {
    input(i, 0) = sqrt(2.0 + 2.0*x1[i]) * cos(2.0*PI*sqrt(2.0+2.0*x1[i]));
    input(i, 1) = sqrt(2.0 + 2.0*x1[i]) * sin(2.0*PI*sqrt(2.0+2.0*x1[i]));
    input(i, 2) = x2[i] * 2.0;
  }
  delete [] x1;
  delete [] x2;
  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}

// Generate a swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_swissroll(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng){
  YA_ColD rand_x=vm_random(sample_size,1,rng);
  YA_ColD rand_y=vm_random(sample_size,1,rng);
  YA_ColD tt = (3.0*PI/2.0)*(1.0+2.0*rand_x);
  YA_ColD height = 21.0*rand_y;
  for (int i=0; i<sample_size; i++){
    if (rand_x(i)<0.1) colors[i] = 1;
    else if (rand_x(i)<0.2) colors[i] = 2;
    else if (rand_x(i)<0.3) colors[i] = 3;
    else if (rand_x(i)<0.4) colors[i] = 4;
    else if (rand_x(i)<0.5) colors[i] = 5;
    else if (rand_x(i)<0.6) colors[i] = 6;
    else if (rand_x(i)<0.7) colors[i] = 7;
    else if (rand_x(i)<0.8) colors[i] = 8;
    else if (rand_x(i)<0.9) colors[i] = 9;
    else if (rand_x(i)<1.0) colors[i] = 10;
  }
  input.setup(tt.numel(),3);
  input(":",0) = tt.dot_star(cos(tt));
  input(":",1) = height;
  input(":",2) = tt.dot_star(sin(tt));

  // scale data between 0 and 1 for autoencoder
  input = (input-minimum(input))/(maximum(input)-minimum(input));
}

}
#endif
