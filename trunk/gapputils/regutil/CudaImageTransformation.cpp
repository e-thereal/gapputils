/*!
 * \file CudaImageTransformation.cpp
 * 
 * Copyright (c) 200x by <your name/ organization here>
 */
/*
 * CudaImageTransformation.cpp
 *
 *  Created on: Nov 13, 2008
 *      Author: tombr
 */

#include "CudaImageTransformation.h"

#include <culib/transform.h>
#include "constants.h"

#include <iostream>

using namespace culib;
using namespace std;
using namespace reglib;
using namespace optlib;

namespace regutil {

/*!
 * \brief
 * Write brief comment for CudaImageTransformation here.
 * 
 * \param imgDim
 * Description of parameter imgDim.
 * 
 * \throws <exception class>
 * Description of criteria for throwing this exception.
 * 
 * Write detailed description for CudaImageTransformation here.
 * 
 * \remarks
 * Write remarks for CudaImageTransformation here.
 * 
 * \see
 * Separate items with the '|' character.
 */
CudaImageTransformation::CudaImageTransformation(const dim3& imgDim, const dim3& voxelDim)
 : CudaImage(imgDim, voxelDim), xskip(1), yskip(1), zskip(1)
{
  transformationParameterMapping.resize(DOF);
  for (unsigned i = 0; i < transformationParameterMapping.size(); ++i)
    transformationParameterMapping[i] = -1;
}

void CudaImageTransformation::setTransformation(const ParameterVector& parameter) {
  transformation = parameter;
}

IImageFunction<int3>* CudaImageTransformation::eval(const IImageFunction<int3>& image) {
  const CudaImage* cudaImage;
  if (cudaImage = dynamic_cast<const CudaImage*>(&image)) {

    // Lets go =)
    transform3D(getDevicePointer(), cudaImage->getCudaArray(), getSize(), getMatrix(image), dim3(xskip, yskip, zskip));
  }
  return this;
}

fmatrix4 CudaImageTransformation::getMatrix(const IImageFunction<int3>& image) const {
  const CudaImage* cudaImage;
  if (cudaImage = dynamic_cast<const CudaImage*>(&image)) {

    // The following part looks much more complicated, than it is. The only thing
    // that is done is, to check, whether a possible parameter is given by
    // the transformation vector, by checking whether the transformation parameter
    // mapping provides a valid index. If so, we grab the according parameter.
    // If a parameter is not specified, a default value is used.
    double xtrans = 0, ytrans = 0, ztrans = 0, xrot = 0, yrot = 0, zrot = 0,
           xscale = 1, yscale = 1, zscale = 1;

    if (0 <= transformationParameterMapping[XTrans] &&
        transformationParameterMapping[XTrans] < transformation.size())
    {
      xtrans = transformation[transformationParameterMapping[XTrans]];
    }

    if (0 <= transformationParameterMapping[YTrans] &&
        transformationParameterMapping[YTrans] < transformation.size())
    {
      ytrans = transformation[transformationParameterMapping[YTrans]];
    }

    if (0 <= transformationParameterMapping[ZTrans] &&
        transformationParameterMapping[ZTrans] < transformation.size())
    {
      ztrans = transformation[transformationParameterMapping[ZTrans]];
    }

    if (0 <= transformationParameterMapping[XRot] &&
        transformationParameterMapping[XRot] < transformation.size())
    {
      xrot = transformation[transformationParameterMapping[XRot]];
    }

    if (0 <= transformationParameterMapping[YRot] &&
        transformationParameterMapping[YRot] < transformation.size())
    {
      yrot = transformation[transformationParameterMapping[YRot]];
    }

    if (0 <= transformationParameterMapping[ZRot] &&
        transformationParameterMapping[ZRot] < transformation.size())
    {
      zrot = transformation[transformationParameterMapping[ZRot]];
    }

    if (0 <= transformationParameterMapping[XScale] &&
        transformationParameterMapping[XScale] < transformation.size())
    {
      xscale = transformation[transformationParameterMapping[XScale]];
    }

    if (0 <= transformationParameterMapping[YScale] &&
        transformationParameterMapping[YScale] < transformation.size())
    {
      yscale = transformation[transformationParameterMapping[YScale]];
    }

    if (0 <= transformationParameterMapping[ZScale] &&
        transformationParameterMapping[ZScale] < transformation.size())
    {
      zscale = transformation[transformationParameterMapping[ZScale]];
    }

    return CudaImageTransformation::CreateMatrix(xrot, yrot, zrot, xtrans, ytrans, ztrans,
      xscale, yscale, zscale, cudaImage->getSize(), cudaImage->getVoxelSize(), getSize(), getVoxelSize());
  }
  return make_fmatrix4_identity();
}

void CudaImageTransformation::setXTranslationIndex(int index) {
  transformationParameterMapping[XTrans] = index;
}

void CudaImageTransformation::setYTranslationIndex(int index) {
  transformationParameterMapping[YTrans] = index;
}

void CudaImageTransformation::setZTranslationIndex(int index) {
  transformationParameterMapping[ZTrans] = index;
}

void CudaImageTransformation::setXRotationIndex(int index) {
  transformationParameterMapping[XRot] = index;
}

void CudaImageTransformation::setYRotationIndex(int index) {
  transformationParameterMapping[YRot] = index;
}

void CudaImageTransformation::setZRotationIndex(int index) {
  transformationParameterMapping[ZRot] = index;
}

void CudaImageTransformation::setXScalingIndex(int index) {
  transformationParameterMapping[XScale] = index;
}

void CudaImageTransformation::setYScalingIndex(int index) {
  transformationParameterMapping[YScale] = index;
}

void CudaImageTransformation::setZScalingIndex(int index) {
  transformationParameterMapping[ZScale] = index;
}

void CudaImageTransformation::setParameter(int id, double value) {
  if (id != BlurringSigma)              // Suppress non-sense blurring
    CudaImage::setParameter(id, value);
  switch(id) {
    case XSkip: xskip = value; break;
    case YSkip: yskip = value; break;
    case ZSkip: zskip = value; break;
  }
}

fmatrix4 CudaImageTransformation::CreateMatrix(double xrot, double yrot, double zrot, double xtrans, double ytrans, double ztrans,
      double xscale, double yscale, double zscale, const dim3& inImgDim, const dim3& inVoxelDim, const dim3& outImgDim, const dim3& outVoxelDim)
{
  // Translation
  fmatrix4 translate = make_fmatrix4_translation(xtrans, ytrans, ztrans);

  // Rotations
  fmatrix4 rotateX = make_fmatrix4_rotationX(-xrot * M_PI / 180.0);
  fmatrix4 rotateY = make_fmatrix4_rotationY(-yrot * M_PI / 180.0);
  fmatrix4 rotateZ = make_fmatrix4_rotationZ(-zrot * M_PI / 180.0);

  // Scaling
  fmatrix4 scaling = make_fmatrix4_scaling(xscale, yscale, zscale);

  // Find centers (-0.5, because we rotate around the center of a voxel)
  double infxc = double(inImgDim.x) / 2.0-0.5;
  double infyc = double(inImgDim.y) / 2.0-0.5;
  double infzc = double(inImgDim.z) / 2.0-0.5;

  double outfxc = double(outImgDim.x) / 2.0-0.5;
  double outfyc = double(outImgDim.y) / 2.0-0.5;
  double outfzc = double(outImgDim.z) / 2.0-0.5;

  double inXVoxelSize = double(inVoxelDim.x) * mu2mm;
  double inYVoxelSize = double(inVoxelDim.y) * mu2mm;
  double inZVoxelSize = double(inVoxelDim.z) * mu2mm;

  double outXVoxelSize = double(outVoxelDim.x) * mu2mm;
  double outYVoxelSize = double(outVoxelDim.y) * mu2mm;
  double outZVoxelSize = double(outVoxelDim.z) * mu2mm;

  fmatrix4 moveCenterToOrigin = make_fmatrix4_translation(-outfxc, -outfyc, -outfzc);
  fmatrix4 moveBack = make_fmatrix4_translation(infxc, infyc, infzc);

  // Transfer coordinates the physical space
  fmatrix4 applyDimension = make_fmatrix4_scaling(outXVoxelSize, outYVoxelSize, outZVoxelSize);
  fmatrix4 revertDimension = make_fmatrix4_scaling(1./inXVoxelSize, 1./inYVoxelSize, 1./inZVoxelSize);

  // Calculate the final transformation. Since all transformations are applied
  // to the coordinate in reverse order, we have to do the multiplication
  // in reverse order too, to get the desired transformation
  return moveBack * revertDimension
                  * scaling * rotateZ * rotateY * rotateX * translate
                  * applyDimension * moveCenterToOrigin;
}

}
