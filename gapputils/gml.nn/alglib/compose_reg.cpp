
// This program assumes  the coordinate system stays the  same for the
// first  and second  transformations, which  is not  true if  you run
// mifresamp twice  to apply two transformations. That  is why compose
// applies  the second  transformation first.  This reversal  of order
// makes the composed transformation work with mifresamp properly.

#include "compose_reg.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <vector>

using namespace std;
using namespace alglib;

// constructor
_3D::_3D(void)
{
  pi = M_PI;

  _3DSetObject (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  for (int i = 0; i < 3; ++i) {
    dims1[i] = dims2[i] = 0;
    sizes1[i] = sizes2[i] = 0.0;
  }

  fName1 = "";
  fName2 = "";

  // initialize position
  position[0] = 0.0;
  position[1] = 0.0;
  position[2] = 0.0;
  position[3] = 1.0;
}

void _3D::_3DSetObject(double xTrans, double yTrans, double zTrans,
		       double xAngle, double yAngle, double zAngle)
{
  double tmpMat[12];

  // inverse transform
  MoveFill (InverseObjectMoveMatrix, -xTrans, -yTrans, -zTrans);
  RotateFill (InverseObjectRotationMatrix, xAngle, yAngle, zAngle);
  getTransposeRot (tmpMat, true);

  for (int i = 0; i < 12; ++i)
    InverseObjectRotationMatrix[i] = tmpMat[i];

  MatrixMultiply (InverseObjectMoveMatrix, InverseObjectRotationMatrix, InverseObjectTransform);

  // forward transform
  MoveFill (ObjectMoveMatrix, xTrans, yTrans, zTrans);
  RotateFill (ObjectRotationMatrix, xAngle, yAngle, zAngle);

  // same internal order as mifresamp
  MatrixMultiply (ObjectRotationMatrix, ObjectMoveMatrix, ObjectTransform);
}

void _3D::MatrixMultiply(double *A, double *B, double *C)
{
   C[0] = A[0]*B[0] + A[1]*B[4] + A[2]*B[8];
   C[1] = A[0]*B[1] + A[1]*B[5] + A[2]*B[9];
   C[2] = A[0]*B[2] + A[1]*B[6] + A[2]*B[10];
   C[3] = A[0]*B[3] + A[1]*B[7] + A[2]*B[11] + A[3];

   C[4] = A[4]*B[0] + A[5]*B[4] + A[6]*B[8];
   C[5] = A[4]*B[1] + A[5]*B[5] + A[6]*B[9];
   C[6] = A[4]*B[2] + A[5]*B[6] + A[6]*B[10];
   C[7] = A[4]*B[3] + A[5]*B[7] + A[6]*B[11] + A[7];

   C[8] = A[8]*B[0] + A[9]*B[4] + A[10]*B[8];
   C[9] = A[8]*B[1] + A[9]*B[5] + A[10]*B[9];
   C[10] = A[8]*B[2] + A[9]*B[6] + A[10]*B[10];
   C[11] = A[8]*B[3] + A[9]*B[7] + A[10]*B[11] + A[11];
}

//---------------------------------------------------------------------------
void _3D::MoveFill(double *A, double Cx, double Cy, double Cz)
{
   A[0] = 1;   A[1] = 0;   A[2] = 0;   A[3] = Cx;
   A[4] = 0;   A[5] = 1;   A[6] = 0;   A[7] = Cy;
   A[8] = 0;   A[9] = 0;   A[10]= 1;   A[11]= Cz;
}

//---------------------------------------------------------------------------
void _3D::RotateFill(double *A, double xAngle, double yAngle, double zAngle)
{
   double x[12];
   double y[12];
   double z[12];
   double tempArray[12];
   double cx,cy,cz;
   double sx,sy,sz;

   cx = cos (xAngle);
   cy = cos (yAngle);
   cz = cos (zAngle);

   sx = sin (xAngle);
   sy = sin (yAngle);
   sz = sin (zAngle);

   // use the same matrix as used in mifreg and mifresamp
   // but don't forget that mifreg and mifresamp apply these to the coordinate system
   x[0]=1;     x[1]=0;     x[2] =0;    x[3] =0;
   x[4]=0;     x[5]=cx;    x[6] =sx;   x[7] =0;
   x[8]=0;     x[9]=-sx;   x[10]=cx;   x[11]=0;

   y[0]=cy;    y[1]=0;     y[2] =-sy;  y[3] =0;
   y[4]=0;     y[5]=1;     y[6] =0;    y[7] =0;
   y[8]=sy;    y[9]=0;     y[10]=cy;   y[11]=0;

   z[0]=cz;    z[1]=sz;    z[2] =0;    z[3] =0;
   z[4]=-sz;   z[5]=cz;    z[6] =0;    z[7] =0;
   z[8]=0;     z[9]=0;     z[10]=1;    z[11]=0;

   MatrixMultiply (z, y, tempArray);   // multiply 2 matrices
   MatrixMultiply (tempArray, x, A);   // multiply result by 3rd matrix
}

// copy the rotation component of the transform matrix to the rotation
// matrix
void _3D::copyTransformToRotate ()
{
  for (int i = 0; i < 12; ++i)
    ObjectRotationMatrix[i] = ObjectTransform[i];

  ObjectRotationMatrix[3] = ObjectRotationMatrix[7] = ObjectRotationMatrix[11] = 0.0;

  double tmpMat[12];

  getTransposeRot (tmpMat);

  for (int i = 0; i < 12; ++i)
    InverseObjectRotationMatrix[i] = tmpMat[i];
}

void _3D::applyTransform (bool inverse)
{
  const double* A = inverse ? InverseObjectTransform : ObjectTransform;

  double tempPos[4];

  for (int r = 0; r < 3; ++r) {
    double f = 0.0;

    for (int c = 0; c < 4; ++c) {
      int offset = c + r * 4;
      f += A[offset] * position[c];
    }

    tempPos[r] = f;
  }

  for (int i = 0; i < 3; ++i)
    position[i] = tempPos[i];
}

const double* _3D::getPosition () const
{
  return position;
}

void _3D::setPosition (const double& x, const double& y, const double& z)
{
  position[0] = x;
  position[1] = y;
  position[2] = z;
  position[3] = 1.0;
}

void _3D::printPosition ()
{
  printf ("Position: %f, %f, %f\n", position[0], position[1], position[2]);
}

void _3D::loadObjectIdentity ()
{
  double* A = ObjectTransform;

  A[0] = 1.0;     A[1] = 0.0;    A[2]  = 0.0;   A[3]  = 0.0;
  A[4] = 0.0;     A[5] = 1.0;    A[6]  = 0.0;   A[7]  = 0.0;
  A[8] = 0.0;     A[9] = 0.0;    A[10] = 1.0;   A[11] = 0.0;
}

void _3D::showMatrix (double* A)
{
  if (A == NULL)
    A = ObjectTransform;

  cout << setw (8) << setprecision (6)
       << A[0] << "\t" << A[1] << "\t" << A[2]  << "\t" << A[3]  << endl;
  cout << setw (8) << setprecision (6)
       << A[4] << "\t" << A[5] << "\t" << A[6]  << "\t" << A[7]  << endl;
  cout << setw (8) << setprecision (6)
       << A[8] << "\t" << A[9] << "\t" << A[10] << "\t" << A[11] << endl << endl;
}

// get the transpose of the rotation part only
void _3D::getTransposeRot (double* A, bool inverse)
{
  if (A == NULL) {
    cout << "Cannot put transpose into a NULL matrix." << endl;
    return;
  }

  double* B = inverse ? InverseObjectRotationMatrix : ObjectRotationMatrix;

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      int i = r * 4 + c;
      int i1 = c * 4 + r;

      A[i] = B[i1];
    }
  }

  A[3] = A[7] = A[11] = 0.0;
}

// extraction rotation component  from transformation matrix - adopted
// from Graphics Gems IV
void _3D::extractRotation (double& outX, double& outY, double& outZ, bool inverse)
{
  double* A = inverse ? InverseObjectTransform : ObjectTransform;
  double cy = sqrt (A[0] * A[0] + A[4] * A[4]);

  if (cy > 16.0 * const_tiny) {
    outX = atan2 (A[9], A[10]);
    outY = atan2 (-A[8], cy);
    outZ = atan2 (A[4], A[0]);
  } else {
    outX = atan2 (-A[6], A[5]);
    outY = atan2 (-A[8], cy);
    outZ = 0.0;
  }

  // the extraction  code assumes a right-handed rotation,  so we have
  // to take the negative values
  outX = -outX / M_PI * 180.0;
  outY = -outY / M_PI * 180.0;
  outZ = -outZ / M_PI * 180.0;

  // printf ("%f %f %f\n", outX, outY, outZ);
}

// multiply the transform matrix by the inverse of the rotation matrix
void _3D::extractTranslation (double& outX, double& outY, double& outZ, bool inverse)
{
  double A[12], C[12];
  getTransposeRot (A, inverse);

  if (inverse)
    MatrixMultiply (A, InverseObjectTransform, C);
  else
    MatrixMultiply (A, ObjectTransform, C);

  outX = C[3];
  outY = C[7];
  outZ = C[11];

  // printf ("%f %f %f\n", outX, outY, outZ);
}

void _3D::readTransform (ifstream& inFile)
{
  double xr (0.0), yr (0.0), zr (0.0), xt (0.0), yt (0.0), zt (0.0);

  inFile >> fName1;
  inFile >> dims1[X] >> dims1[Y] >> dims1[Z];
  inFile >> sizes1[X] >> sizes1[Y] >> sizes1[Z];

  inFile >> fName2;
  inFile >> dims2[X] >> dims2[Y] >> dims2[Z];
  inFile >> sizes2[X] >> sizes2[Y] >> sizes2[Z];

  inFile >> xr >> yr >> zr >> xt >> yt >> zt;

  // convert to radians
  xr = xr / 180.0 * M_PI;
  yr = yr / 180.0 * M_PI;
  zr = zr / 180.0 * M_PI;

  if (inFile.fail ()) {
    cout << "Failed to read transformation parameters." << endl;
    exit (-1);
  } else
    _3DSetObject (xt, yt, zt, xr, yr, zr);
}

void _3D::readMatrix (ifstream& inFile)
{
  for (int i = 0; i < 12; ++i)
    inFile >> ObjectTransform[i];

  if (inFile.fail ()) {
    cout << "Failed to read transformation matrix." << endl;
    exit (-1);
  }
}

void _3D::composeTransform (_3D* inObj, bool inverse)
{
  if (inObj == NULL) {
    cout << "ERROR: NULL object in _3D::composeTransform.";
    return;
  }

  double tmpMat[12];

  if (inverse)
    MatrixMultiply (ObjectTransform, inObj -> InverseObjectTransform, tmpMat);
  else
    MatrixMultiply (ObjectTransform, inObj -> ObjectTransform, tmpMat);
    // MatrixMultiply (inObj -> ObjectTransform, ObjectTransform, tmpMat);

  for (int i = 0; i < 12; ++i)
    ObjectTransform[i] = tmpMat[i];
}

bool invert_mode (false);

ALGLIB_API void alglib::compose (const vector<double>& transform, const vector<double>& transform1,
	      vector<double>& outTransform)
{
  if (transform.size () < 6 || transform1.size () < 6) {

    cout << "Transformation does not have enough parameters in compose ()" << endl;
    return;
  }

  _3D* currObject = new _3D;

  vector<double> inTrans (6), inTrans1 (6);

  // convert to radians
  for (int i = 0; i < 6; ++i) {

    inTrans[i] = transform[i] / 180.0 * M_PI;
    inTrans1[i] = transform1[i] / 180.0 * M_PI;
  }

  currObject->_3DSetObject (inTrans[3], inTrans[4], inTrans[5],
			    inTrans[0], inTrans[1], inTrans[2]);

  _3D* currObject1 = new _3D;

  currObject1->_3DSetObject (inTrans1[3], inTrans1[4], inTrans1[5],
			     inTrans1[0], inTrans1[1], inTrans1[2]);

  // compose the two transformations
  currObject -> composeTransform (currObject1);

  currObject -> extractRotation (outTransform[0], outTransform[1], outTransform[2]);
  // need the proper rotation matrix to extract translation
  currObject -> copyTransformToRotate ();
  currObject -> extractTranslation (outTransform[3], outTransform[4], outTransform[5]);

  delete currObject;
  delete currObject1;
}

ALGLIB_API void alglib::invert (const vector<double>& transform, vector<double>& outTransform)
{
  if (transform.size () < 6) {

    cout << "Transformation does not have enough parameters in invert ()" << endl;
    return;
  }

  _3D* currObject = new _3D;

  vector<double> inTrans (6);

  // convert to radians
  for (int i = 0; i < 6; ++i)
    inTrans[i] = transform[i] / 180.0 * M_PI;

  currObject->_3DSetObject (inTrans[3], inTrans[4], inTrans[5],
 			    inTrans[0], inTrans[1], inTrans[2]);

  // invert the transformation
  currObject -> extractRotation (outTransform[0], outTransform[1], outTransform[2], true);
  currObject -> extractTranslation (outTransform[3], outTransform[4], outTransform[5], true);

  delete currObject;
}
