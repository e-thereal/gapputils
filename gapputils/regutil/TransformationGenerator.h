/**
 * @file TransformationGenerator.h
 * @brief Contains the TransformationGenerator class
 *
 * @date Nov 13, 2008
 * @author Tom Brosch
 */

#ifndef _REGUTIL_TRANSFORMATIONGENERATOR_H_
#define _REGUTIL_TRANSFORMATIONGENERATOR_H_

#include <reglib/RegistrationProblem.h>

namespace regutil {

/// This class can create an TransformationGenerator out of an CudaImageTransformation
/**
 * @remarks
 * - Please make sure, that the type parameter is a descendant of CudaImageTransformation
 */

template<class T>
class TransformationGenerator : public T,
                                public virtual reglib::ITransformationGenerator<int3>
{
public:
  /// Constructor to create a new TransformationGenerator instance
  /**
   * @param[in] imgDim The dimension of images, that can be transformed by the image transformation
   */
  TransformationGenerator(const dim3& imgDim, const dim3& voxelDim) : T(imgDim, voxelDim) { }
  TransformationGenerator(const ICudaImage* referenceImage) : T(referenceImage) { }

  /// Evaluating a transformation generator means creating a new image transformation
  /**
   * @param[in] parameter The parameter vector, which describes the new image transformation
   *
   * @return The new image transformation, that performs transformations according to the given parameter vector
   */
  virtual reglib::IImageTransformation<int3>* eval(const reglib::ParameterVector& parameter) {
    T::setTransformation(parameter);
    return this;
  }
};

}

#endif /* _MRLIB_TRANSFORMATIONGENERATOR_H_ */
