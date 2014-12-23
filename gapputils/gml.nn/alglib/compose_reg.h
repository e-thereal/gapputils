/**
 * @file compose_reg.h
 * @brief Basic functions to compose transformation parameters
 *
 * @date Nov 27, 2008
 * @author Unknown
 */

#ifndef _ALGLIB__3D_H_
#define _ALGLIB__3D_H_

#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "alglib.h"

namespace alglib {

const int X = 0;
const int Y = 1;
const int Z = 2;

//---------------------------------------------------------------------------
/// This class is not intended to be used externally
class ALGLIB_API _3D {
 private:
  double position[4]; // current position

   // internal vectors and matrices
  double ObjectMoveMatrix[12];
  double ObjectRotationMatrix[12];
  double ObjectTransform[12];
  double InverseObjectMoveMatrix[12];
  double InverseObjectRotationMatrix[12];
  double InverseObjectTransform[12];

  // helper functions
  void MatrixMultiply (double *A, double *B, double *C);
  void MoveFill (double *A, double Cx, double Cy, double Cz);
  void RotateFill (double *A, double Cx, double Cy, double Cz);

 public:
  double pi;
  std::string fName1, fName2;
  int dims1[3];
  int dims2[3];
  double sizes1[3];
  double sizes2[3];

  _3D (void); // constructor

  void _3DSetObject (double, double, double, double, double, double); // set the transform

  void readTransform (std::ifstream&);
  void readMatrix (std::ifstream&);
  void applyTransform (bool = false);
  void showMatrix (double* = 0);
  void getTransposeRot (double*, bool = false);
  void extractRotation (double&, double&, double&, bool = false);
  void copyTransformToRotate ();
  void extractTranslation (double&, double&, double&, bool = false);
  void composeTransform (_3D*, bool = false);

  const double* getPosition () const;
  void setPosition (const double&, const double&, const double&);
  void printPosition ();
  void loadObjectIdentity ();
};

/// This class is not intended to be used externally
template<class T>
class ALGLIB_API myVector : public std::vector<T> {
public:

  myVector () {};

  myVector (int i) : std::vector<T> (i) {
    fill (this->begin (), this->end (), 0.0);
  };

  myVector (int i, double f) : std::vector<T> (i, f) {};

  ~myVector () {};

  template<class U>
  friend std::ostream& operator<< (std::ostream&, const myVector<U>&);
};

/// Prints a myVector to an output stream
template <class T> std::ostream& operator<< (std::ostream& os, const myVector<T>& v)
{
  int size = v.size ();

  for (int i = 0; i < size; ++i) {

    os << v[i];

    if (i < size - 1)
      os << " ";
  }

  os << std::endl;

  return os;
}

/// Buffer size for internal use
const int char_buffer_size = 300;

/// This is really tiny
const double const_tiny = 1.0E-6;

/// Compose two transformations
/**
 * Use this function to get the resulting transformation out of two given transformations
 *
 * @param[in]  transform    The first transformation
 * @param[in]  transform1   The second transformation
 * @param[out] outTransform The resulting transformation
 */
ALGLIB_API void compose (const std::vector<double>& transform,
    const std::vector<double>& transform1, std::vector<double>& outTransform);

/// Computes the inverse of a given transformation
/**
 * @param[in]  transform    The input transformation
 * @param[out] outTransform The inverse transformation
 */
ALGLIB_API void invert (const std::vector<double>& transform, std::vector<double>& outTransform);

}

#endif /* _ALGLIB__3D_H_ */
