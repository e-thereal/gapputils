/**
 * @file IParameterizable.h
 * @brief The IParameterizable interface
 *
 * @date Nov 12, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IPARAMETERIZABLE_H_
#define _OPTLIB_IPARAMETERIZABLE_H_

#include "optlib.h"

namespace optlib {

/// IDs for all parameters, used for the optimization
enum Parameter {
  XSkip,                ///< Skipping of voxels along the x axis
  YSkip,                ///< Skipping of voxels along the y axis
  ZSkip,                ///< Skipping of voxels along the z axis
  FreeCaches,           ///< Not a real parameter. Forces all caches to be clean up
  LineTolerance,        ///< Tolerance of the line optimization step
  BlurringSigma,        ///< Sigma of the image blurring
  HistogramSigma,       ///< Sigma of the histogram blurring
  Tolerance,            ///< Overall tolerance of the optimization algorithm
  SimilarityMode,       ///< Specifies the similarity measure, that should be used
  SimilarityMethod,     ///< Specifies the method used to calculate the similarity measure
  LineMin,              ///< The lower bound of the search interval of the line optimization
  LineMax,              ///< The upper bound of the search interval of the line optimization
  InitialStepSize,      ///< The initial step size used for the line optimization
  ResetWorkingCopy,     ///< Not a real parameter. Forces a reset of the working copy.
  MinFrom,              ///< For adjusting pixel values; sets parameter
  MaxFrom,              ///< For adjusting pixel values; sets parameter
  MinTo,                ///< For adjusting pixel values; sets parameter
  MaxTo,                ///< For adjusting pixel values; sets parameter and triggers the adjustment
  Scaling,              ///< Scales an image. The parameter is set as an float3*
  GridSampling,         ///< Sets the parameter for the grid sampling. (GridParameter*)
  UserParameter = 100   ///< Use this value as a starting point for user defined parameters
};

/// Tells, that a given class can be parameterized during the optimization
class OPTLIB_API IParameterizable {
public:

  /// Sets a parameter
  /**
   * @param[in] id    The unique ID of the parameter
   * @param[in] value The parameter value
   */
  virtual void setParameter(int id, double value) = 0;

  /// Sets a parameter
  /**
   * @param[in] id    The unique ID of the parameter
   * @param[in] value A pointer to an object containing the parameter
   */
  virtual void setParameter(int id, void* value) = 0;
};

}

#endif /* _OPTLIB_IPARAMETERIZABLE_H_ */
