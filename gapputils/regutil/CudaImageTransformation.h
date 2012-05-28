/**
 * @file CudaImageTransformation.h
 * @brief The CudaImageTransformation class
 *
 * @date Nov 13, 2008
 * @author Tom Brosch
 */

#ifndef _REGUTIL_CUDAIMAGETRANSFORMATION_H_
#define _REGUTIL_CUDAIMAGETRANSFORMATION_H_

#include "CudaImage.h"
#include <optlib/IParameterizable.h>
#include <reglib/RegistrationProblem.h>

#include <vector>

#include "regutil.h"

namespace regutil {

/// Applies a predefined transformation to an image function
/**
 * @remarks
 * - You can apply rotations, translations and scaling to an image in 3 dimensions
 * - Set the transformation parameter before you call eval() with the
 *   setTransformation() method
 * - There are 9 parameters of the transformation: rotation around the x, y and z axis,
 *   translation along the x, y and z axis and scaling along the x, y and z axis.
 *   In order to get everything right, you have to tell the function the position
 *   of each parameter (the index) in the transformation vector.
 */
class CudaImageTransformation : public CudaImage,
                                public virtual reglib::IImageTransformation<int3>,
                                public virtual optlib::IParameterizable
{
private:
  reglib::ParameterVector transformation;
  int xskip, yskip, zskip;

  enum TransformationParameter {XTrans = 0, YTrans, ZTrans, XRot, YRot, ZRot,
    XScale, YScale, ZScale, DOF};

  /// The transformation parameter mapping tells us, which transformation parameter
  /// can be found at with index in the transformation vector.
  /// So if I want to know the index of the translation in x direction,
  /// transformationParameterMapping[XTrans] tells us the index at which I can
  /// find the wanted information in the transformation vector. Thereby, an index
  /// of -1 tells us, that the according parameter is not specified by the
  /// transformation vector
  std::vector<int> transformationParameterMapping;

public:

  /// Constructor to set up the image transformation
  /**
   * This method will allocate memory for the transformed image
   * @param[in] imgDim The width, height and depth of the image, that should be transformed
   */
  CudaImageTransformation(const dim3& imgDim, const dim3& voxelDim);

  /// Define the transformation by a parameter vector
  /**
   * @param[in] parameter The transformation parameters
   */
  void setTransformation(const reglib::ParameterVector& parameter);

  fmatrix4 getMatrix(const reglib::IImageFunction<int3>& image) const;

  /// Apply the transformation to a given image function
  /**
   * This function writes the transformed image to a another ICudaImage and returns
   * a pointer to it.
   *
   * @remarks
   * - The memory, that is used to store the transformed image, was allocated by
   *   the Constructor of this class and will be freed by the Destructor
   *
   * @param[in] image The input image
   *
   * @return A pointer to the transformed ICudaImage
   */
  virtual reglib::IImageFunction<int3>* eval(const reglib::IImageFunction<int3>& image);

  void setXTranslationIndex(int index);   ///< Define the index of the parameter vector that corresponds to the x translation
  void setYTranslationIndex(int index);   ///< Define the index of the parameter vector that corresponds to the y translation
  void setZTranslationIndex(int index);   ///< Define the index of the parameter vector that corresponds to the z translation

  void setXRotationIndex(int index);      ///< Define the index of the parameter vector that corresponds to the x rotation
  void setYRotationIndex(int index);      ///< Define the index of the parameter vector that corresponds to the y rotation
  void setZRotationIndex(int index);      ///< Define the index of the parameter vector that corresponds to the z rotation

  void setXScalingIndex(int index);       ///< Define the index of the parameter vector that corresponds to the x scaling
  void setYScalingIndex(int index);       ///< Define the index of the parameter vector that corresponds to the y scaling
  void setZScalingIndex(int index);       ///< Define the index of the parameter vector that corresponds to the z scaling

  virtual void setParameter(int id, double value);

  static fmatrix4 CreateMatrix(double xrot, double yrot, double zrot, double xtrans, double ytrans, double ztrans,
      double xscale, double yscale, double zscale, const dim3& inImgDim, const dim3& inVoxelDim, const dim3& outImgDim, const dim3& outVoxelDim);
};

}

#endif /* _REGUTIL_CUDAIMAGETRANSFORMATION_H_ */
