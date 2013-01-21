/**
 * @file math3d.h
 * @brief Basic linear algebra for both CPU and GPU
 *
 * @date Sep 16, 2008
 * @author Tom Brosch
 */

#ifndef GML_MATH3D_H_
#define GML_MATH3D_H_

#include <cstdlib>
#include <cmath>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace gml {

struct float4 {
  float x, y, z, w;
};

struct fmatrix4 {
  union {
    struct { float4 r1, r2, r3, r4; };
    float _array[16];
  };
};

__device__ __host__ 
inline float4 make_float4(const float& x, const float& y, const float& z, const float& w) {
  float4 ret;
  ret.x = x;
  ret.y = y;
  ret.z = z;
  ret.w = w;
  return ret;
}

__device__ __host__ 
inline float dot(const float4& x, const float4& y) {
  return x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
}

// fmatrix4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
__device__ __host__ 
inline fmatrix4 make_fmatrix4(float4 r1, float4 r2, float4 r3, float4 r4) {
  fmatrix4 fm4;
  fm4.r1 = r1;
  fm4.r2 = r2;
  fm4.r3 = r3;
  fm4.r4 = r4;
  return fm4;
}

__device__ __host__ 
inline fmatrix4 make_fmatrix4(float m11, float m12, float m13, float m14,
    float m21, float m22, float m23, float m24,
    float m31, float m32, float m33, float m34,
    float m41, float m42, float m43, float m44) {
  fmatrix4 fm4;
  fm4.r1 = make_float4(m11, m12, m13, m14);
  fm4.r2 = make_float4(m21, m22, m23, m24);
  fm4.r3 = make_float4(m31, m32, m33, m34);
  fm4.r4 = make_float4(m41, m42, m43, m44);
  return fm4;
}

// column getter
__device__ __host__ 
inline float4 get_x_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.x, fm4.r2.x, fm4.r3.x, fm4.r4.x);
}

__device__ __host__ 
inline float4 get_y_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.y, fm4.r2.y, fm4.r3.y, fm4.r4.y);
}

__device__ __host__ 
inline float4 get_z_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.z, fm4.r2.z, fm4.r3.z, fm4.r4.z);
}

__device__ __host__ 
inline float4 get_w_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.w, fm4.r2.w, fm4.r3.w, fm4.r4.w);
}

// special getter for float4
__device__ __host__ 
inline float get_x(float4 f) {
  return f.x/f.w;
}

__device__ __host__ 
inline float get_y(float4 f) {
  return f.y/f.w;
}

__device__ __host__ 
inline float get_z(float4 f) {
  return f.z/f.w;
}

// special matrices
__device__ __host__ 
inline fmatrix4 make_fmatrix4_identity() {
  return make_fmatrix4(1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1);
}

__device__ __host__ 
inline fmatrix4 make_fmatrix4_translation(float x = 0, float y = 0, float z = 0) {
  return make_fmatrix4(1, 0, 0, x,
                       0, 1, 0, y,
                       0, 0, 1, z,
                       0, 0, 0, 1);
}

__device__ __host__ 
inline fmatrix4 make_fmatrix4_scaling(float x = 1, float y = 1, float z = 1) {
  return make_fmatrix4(x, 0, 0, 0,
                       0, y, 0, 0,
                       0, 0, z, 0,
                       0, 0, 0, 1);
}

__device__ __host__ 
inline fmatrix4 make_fmatrix4_rotationX(float angle) {
  return make_fmatrix4( 1,          0,           0, 0,
                        0, cos(angle), -sin(angle), 0,
                        0, sin(angle),  cos(angle), 0,
                        0,          0,           0, 1);
}

__device__ __host__ 
inline fmatrix4 make_fmatrix4_rotationY(float angle) {
  return make_fmatrix4(  cos(angle), 0, sin(angle), 0,
                                  0, 1,          0, 0,
                        -sin(angle), 0, cos(angle), 0,
                                  0, 0,          0, 1);
}

__device__ __host__ 
inline fmatrix4 make_fmatrix4_rotationZ(float angle) {
  return make_fmatrix4( cos(angle), -sin(angle), 0, 0,
                        sin(angle),  cos(angle), 0, 0,
                                 0,           0, 1, 0,
                                 0,           0, 0, 1);
}

}

// addition
__device__ __host__ 
inline gml::float4 operator+(const gml::float4& a, const gml::float4& b) {
  return gml::make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __host__ 
inline gml::fmatrix4 operator+(const gml::fmatrix4& a, const gml::fmatrix4& b) {
  return gml::make_fmatrix4(a.r1+b.r1, a.r2+b.r2, a.r3+b.r3, a.r4+b.r4);
}

// multiplication
__device__ __host__ 
inline gml::float4 operator*(const gml::fmatrix4& a, const gml::float4& b) {
  return gml::make_float4(dot(a.r1,b), dot(a.r2,b), dot(a.r3,b), dot(a.r4,b));
}

__device__ __host__ 
inline gml::fmatrix4 operator*(const gml::fmatrix4& a, const gml::fmatrix4& b) {
  gml::float4 xcol = gml::get_x_column(b);
  gml::float4 ycol = gml::get_y_column(b);
  gml::float4 zcol = gml::get_z_column(b);
  gml::float4 wcol = gml::get_w_column(b);
  return gml::make_fmatrix4(gml::dot(a.r1, xcol), gml::dot(a.r1, ycol), gml::dot(a.r1, zcol), gml::dot(a.r1, wcol),
                       gml::dot(a.r2, xcol), gml::dot(a.r2, ycol), gml::dot(a.r2, zcol), gml::dot(a.r2, wcol),
                       gml::dot(a.r3, xcol), gml::dot(a.r3, ycol), gml::dot(a.r3, zcol), gml::dot(a.r3, wcol),
                       gml::dot(a.r4, xcol), gml::dot(a.r4, ycol), gml::dot(a.r4, zcol), gml::dot(a.r4, wcol));
}

#endif /* GML_MATH3D_H_ */
