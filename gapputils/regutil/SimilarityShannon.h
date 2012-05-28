/**
 * @file SimilarityShannon.h
 * @brief Contains the SimilarityShannon class
 *
 * @date Nov 13, 2008
 * @author Tom Brosch
 */

#ifndef _REGUTIL_SIMILARITYSHANNON_H_
#define _REGUTIL_SIMILARITYSHANNON_H_

#include <reglib/RegistrationProblem.h>
#include <optlib/IParameterizable.h>

#include <culib/similarity.h>

#include "regutil.h"

namespace regutil {

/// Computes Mutual Information or Joint Entropy of two images
/**
 * Evaluating this function computes Mutual Information or Joint Entropy
 */
class SimilarityShannon : public virtual reglib::ISimilarityMeasure<int3>,
                          public virtual optlib::IParameterizable
{
private:
  culib::SimilarityConfig simConfig;    ///< A handle to the similarity configuration
  int xskip,                            ///< Every xskip voxel along the x axis is used for the calculation
      yskip,                            ///< Every yskip voxel along the y axis is used for the calculation
      zskip;                            ///< Every zskip voxel along the z axis is used for the calculation

public:

  /// Constructor to create a new SimilarityShannon instance
  /**
   * @param[in] binCount The number of bins, that should be used to calculate histograms
   * @param[in] binScale The number of intensity values, which are mapped to the same bin
   */
  SimilarityShannon(dim3 binCount, float2 binScale, culib::SimilarityConfig::Method method = culib::SimilarityConfig::NewMethod);

  /// Virtual Destructor
  virtual ~SimilarityShannon();

  virtual double eval(const reglib::ImagePair<int3>& images);
  virtual void setParameter(int id, double value);
  virtual void setParameter(int id, void* value);
};

}

#endif /* _REGUTIL_SIMILARITYSHANNON_H_ */
